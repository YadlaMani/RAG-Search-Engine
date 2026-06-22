from cli.hybrid_search_lib.hybrid_search import *
import json
import os
from dotenv import load_dotenv
from google import genai
import time
from sentence_transformers import CrossEncoder


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


def rrf_search(query, k=60, limit=5, rerank_method=None, verbose=True):
    with open("./data/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    hybrid_search = HybridSearch(documents)
    original_limit = limit
    if rerank_method:
        limit = limit * 5
    results = hybrid_search.rrf_search(query, k, limit)
    if verbose:
        if rerank_method:
            print("RFF Search Results before re-ranking")
            for i, (_, res) in enumerate(results):
                print(
                    f"{i + 1}. {res['title']}\nRRF Score: {res['rrf']:.4f}\nBM25 Rank: {res['bm25']:.4f}, Semantic Rank: {res['semantic']:.4f}\n{res['description']}\n"
                )
            rerank_results(original_limit, results, rerank_method, query, k, verbose)
        else:
            for i, (_, res) in enumerate(results):
                print(
                    f"{i + 1}. {res['title']}\nRRF Score: {res['rrf']:.4f}\nBM25 Rank: {res['bm25']:.4f}, Semantic Rank: {res['semantic']:.4f}\n{res['description']}\n"
                )
    return results


def enhance_text(query, enhance=None):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    print(f"Original Query: {query}")
    if enhance:
        contents = ""
        match enhance:
            case "spell":
                contents = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query}"
                """
            case "rewrite":
                contents = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep the rewritten query concise (under 10 words)
                    - It should be a Google-style search query, specific enough to yield relevant results
                    - Don't use boolean logic

                    Examples:
                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                    If you cannot improve the query, output the original unchanged.
                    Output only the rewritten query text, nothing else.

                    User query: "{query}"
                    """
            case "expand":
                contents = f"""Expand the user-provided movie search query below with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    Output only the additional terms; they will be appended to the original query.

                    Examples:
                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"

                    User query: "{query}"
                    """
        res = client.models.generate_content(model="gemma-4-31b-it", contents=contents)

        if res.text:
            print(f"Enhanced query ({enhance}): '{query}' -> '{res.text}'\n")
            return res.text
    return query


def rerank_results(limit, results, method, query, k, verbose=True):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    match method:
        case "individual":
            for _, res in results:
                time.sleep(3)
                contents = f"""Rate how well this movie matches the search query.

                            Query: "{query}"
                            Movie: {res["title"]} - {res["description"]}

                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness

                            Rate 0-10 (10 = perfect match).
                            Output ONLY the number in your response, no other text or explanation.

                            Score:"""
                response = client.models.generate_content(
                    model="gemma-4-31b-it", contents=contents
                )
                text = response.text
                res["rerank"] = float(text.strip()) if text else 0.0
                sorted_results = sorted(
                    results, key=lambda x: x[1].get("rerank", 0.0), reverse=True
                )
                final_results = sorted_results[:limit]
        case "batch":
            movies_list = "\n".join(
                f"ID {doc_id}: {res['title']} - {res['description']}"
                for doc_id, res in results
            )
            contents = f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    Movies:
                    {movies_list}

                    Return the movie IDs in order of relevance, best match first.

                    Your response must be a raw JSON array of the IDs exactly as shown above.
                    Do not wrap the JSON in Markdown. Do not use a ```json code block.
                    Do not include any explanatory text.

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:"""
            response = client.models.generate_content(
                model="gemma-4-31b-it", contents=contents
            )
            text = response.text
            if not text:
                return results[:limit]
            ranked_ids = json.loads(text.strip())
            results_by_id = {doc_id: res for doc_id, res in results}
            reranked = []
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if doc_id in results_by_id:
                    results_by_id[doc_id]["rerank"] = rank
                    reranked.append((doc_id, results_by_id[doc_id]))
            final_results = reranked[:limit]
        case "cross_encoder":
            pairs = []
            for _, res in results:
                pairs.append([query, f"{res['title']} - {res['description']}"])
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            idx = 0
            for _, res in results:
                res["cross-encoder"] = scores[idx]
                idx += 1
            sorted_results = sorted(
                results, key=lambda x: x[1].get("cross-encoder", 0.0), reverse=True
            )
            final_results = sorted_results[:limit]
    if verbose:
        print(
            f"Re-ranking top {limit} results using {method} method...\nReciprocal Rank Fusion Results for {query} (k={k}):"
        )
        for i, (_, res) in enumerate(final_results):
            if method == "batch":
                rerank_line = (
                    f"Re-rank Rank: {res['rerank']}\n" if "rerank" in res else ""
                )
            elif method == "individual":
                rerank_line = (
                    f"Re-rank Score: {res['rerank']:.4f}/10\n"
                    if "rerank" in res
                    else ""
                )
            else:
                rerank_line = (
                    f"Cross Encoder Score: {res['cross-encoder']:.4f}\n"
                    if "cross-encoder" in res
                    else ""
                )
            print(
                f"{i + 1}. {res['title']}\n{rerank_line}RRF Score: {res['rrf']:.4f}\nBM25 Rank: {res['bm25']:.4f}, Semantic Rank: {res['semantic']:.4f}\n{res['description']}\n"
            )
    return final_results


def evaluate_results(query, results):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    contents = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(f"{i + 1}. {res['title']} - {res['description'][:200]}" for i, (_, res) in enumerate(results))}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers other than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""
    resp = client.models.generate_content(model="gemma-4-31b-it", contents=contents)
    text = resp.text
    if not text:
        print("No response from model")
        return
    scores = json.loads(text.strip())
    for i, (_, res) in enumerate(results):
        print(f"{i + 1}. {res['title']}: {scores[i]}/3")
