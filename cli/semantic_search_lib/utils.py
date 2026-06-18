from cli.semantic_search_lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
)
import json


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    print(f"Text: {text}")
    embedding = semantic_search.generate_embedding(text)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query.strip())
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def search_movies(query, limit):
    semantic_search = SemanticSearch()
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, res in enumerate(results):
        print(f"{i + 1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description']}")


def chunk_text(text, limit, overlap):
    chunks = text.strip().split(" ")
    cnt = 1
    i = 0
    res = []
    while i < len(chunks):
        batch = chunks[i : i + limit]
        text = " ".join(batch)
        res.append(text)
        i = i + limit - overlap
        cnt += 1
    return res


def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    chunk_embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(
        documents
    )
    return chunked_semantic_search


def search_chunked(query, limit):
    chunked_semantic_search = embed_chunks()
    return chunked_semantic_search.search_chunks(query, limit)
