from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from cli.semantic_search_lib.chunking import semantic_chunk
import pickle
import json
import os
from sentence_transformers import CrossEncoder


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    CACHE_DIR = Path("/Users/mani/Developer/bootdev/rag-search-engine/cache")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        Path("cache").mkdir(exist_ok=True)
        self.embeddings_path = Path(
            os.path.join(self.CACHE_DIR, "movie_embeddings.npy")
        )

    def build_embeddings(self, documents):
        self.documents = documents
        movies_list = []
        for doc_id, doc in enumerate(documents):
            self.document_map[doc_id] = doc
            movies_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies_list, show_progress_bar=True)
        self.CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                self.documents = documents
                self.document_map = {i: doc for i, doc in enumerate(documents)}
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embeddings = self.generate_embedding(query)
        results = []
        for id, embeddings in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embeddings, embeddings)
            results.append(
                {
                    "score": similarity,
                    "title": self.document_map[id]["title"],
                    "description": self.document_map[id]["description"],
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = Path(
            os.path.join(self.CACHE_DIR, "chunk_embeddings.npy")
        )
        self.chunk_metadata_path = Path(
            os.path.join(self.CACHE_DIR, "chunk_metadata.pkl")
        )

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunk_metadata = []
        all_chunks = []
        for doc_id, doc in enumerate(documents):
            self.document_map[doc_id] = doc
            if not doc["description"].strip():
                continue
            chunks = semantic_chunk(doc["description"], 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": doc_id,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "wb") as file:
            pickle.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, file
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "rb") as f:
                data = pickle.load(f)
                self.chunk_metadata = data["chunks"]
            self.documents = documents
            self.document_map = {i: doc for i, doc in enumerate(documents)}
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query = query.strip()
        if query == "":
            return []
        query_embeddings = self.generate_embedding(query)
        if self.chunk_metadata is None or self.chunk_embeddings is None:
            raise ValueError(
                "Call build_chunk_embeddings the chunks_embeddings and chunk_metadata is not build yet"
            )
        chunk_scores = []

        for idx, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embeddings, chunk)
            meta = self.chunk_metadata[idx]
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": score,
                }
            )
        movies_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movies_scores.__contains__(movie_idx):
                movies_scores[movie_idx] = max(movies_scores[movie_idx], score)
            else:
                movies_scores[movie_idx] = score
        sorted_movies_scores = dict(
            sorted(movies_scores.items(), key=lambda x: x[1], reverse=True)
        )
        result = []
        for idx, score in sorted_movies_scores.items():
            result.append(
                {
                    "id": idx,
                    "title": self.document_map[idx]["title"],
                    "document": self.document_map[idx]["description"][:100],
                    "score": round(score, 4),
                }
            )
            if len(result) == limit:
                return result
        return result
