from cli.hybrid_search_lib.hybrid_search import *
import json


def normalize(scores):
    if len(scores) == 0:
        return
    maxi = max(scores)
    mini = min(scores)
    if maxi == mini:
        for _ in scores:
            print(f"* 1.0000")
    else:
        for score in scores:
            print(f"* {(score - mini) / (maxi - mini):.4f}")


def weighted_search(query, alpha, limit):
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, (doc_id, res) in enumerate(results):
        print(
            f"{i + 1}. {res['title']}\nHybrid Score: {res['hybrid']:.4f}\nBM25: {res['bm25']:.4f}, Semantic: {res['semantic']:.4f}\n{res['description']}\n"
        )


def rrf_search(query, k, limit):
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, (doc_id, res) in enumerate(results):
        print(
            f"{i + 1}. {res['title']}\nRRF Score: {res['rrf']:.4f}\nBM25 Rank: {res['bm25']:.4f}, Semantic Rank: {res['semantic']:.4f}\n{res['description']}\n"
        )
