import os

from cli.semantic_search_lib.semantic_search import ChunkedSemanticSearch
from cli.keyword_search_lib.inverted_index import InvertedIndex


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        semantic_search_result = self.semantic_search.search_chunks(query, 500 * limit)
        keyword_search_result = self._bm25_search(query, 500 * limit)

        kw_scores = list(keyword_search_result.values())
        kw_min, kw_max = (min(kw_scores), max(kw_scores)) if kw_scores else (0, 1)
        normalized_keyword_scores = {
            doc_id: (1.0 if kw_max == kw_min else (score - kw_min) / (kw_max - kw_min))
            for doc_id, score in keyword_search_result.items()
        }
        semantic_scores = {item["id"]: item["score"] for item in semantic_search_result}
        sem_vals = list(semantic_scores.values())
        sem_min, sem_max = (min(sem_vals), max(sem_vals)) if sem_vals else (0, 1)
        normalized_semantic_scores = {
            doc_id: (
                1.0 if sem_max == sem_min else (score - sem_min) / (sem_max - sem_min)
            )
            for doc_id, score in semantic_scores.items()
        }

        all_doc_ids = set(normalized_keyword_scores) | set(normalized_semantic_scores)
        results = {}
        for doc_id in all_doc_ids:
            hybrid_score = (alpha * normalized_keyword_scores.get(doc_id, 0.0)) + (
                (1 - alpha) * normalized_semantic_scores.get(doc_id, 0.0)
            )
            results[doc_id] = {
                "hybrid": hybrid_score,
                "bm25": normalized_keyword_scores.get(doc_id, 0.0),
                "semantic": normalized_semantic_scores.get(doc_id, 0.0),
                "title": self.semantic_search.document_map[doc_id]["title"],
                "description": self.semantic_search.document_map[doc_id]["description"],
            }
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["hybrid"], reverse=True
        )
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        semantic_search_result = self.semantic_search.search_chunks(query, 500 * limit)
        keyword_search_result = self._bm25_search(query, 500 * limit)
        semantic_ranks = {}
        keyword_ranks = {}
        for i, result in enumerate(semantic_search_result):
            semantic_ranks[result["id"]] = i + 1
        for i, result in enumerate(keyword_search_result):
            keyword_ranks[result] = i + 1
        results = {}
        all_doc_ids = set(keyword_ranks) | set(semantic_ranks)
        for doc_id in all_doc_ids:
            sem_rank = semantic_ranks.get(doc_id)
            kw_rank = keyword_ranks.get(doc_id)
            rff_score = (1 / (k + sem_rank) if sem_rank else 0) + (
                1 / (k + kw_rank) if kw_rank else 0
            )
            results[doc_id] = {
                "rrf": rff_score,
                "bm25": keyword_ranks.get(doc_id, 0),
                "semantic": semantic_ranks.get(doc_id, 0),
                "title": self.semantic_search.document_map[doc_id]["title"],
                "description": self.semantic_search.document_map[doc_id]["description"],
            }
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["rrf"], reverse=True
        )
        return sorted_results[:limit]
