from cli.hybrid_search_lib.hybrid_search import *
import json
import os
from dotenv import load_dotenv
from google import genai


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


def enhance_text(query, enhance):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
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
