import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.hybrid_search_lib.utils import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit
    with open("./data/golden_dataset.json", "r") as f:
        data = json.load(f)
    testcases = data["test_cases"]
    print(f"k={limit}")

    for testcase in testcases:
        results = rrf_search(
            enhance_text(testcase["query"]), limit=limit, verbose=False
        )
        retrieved = [res["title"] for (_, res) in results]
        relevant_retrieved = set(testcase["relevant_docs"]) & set(retrieved)
        precision = len(relevant_retrieved) / limit
        recall = len(relevant_retrieved) / len(testcase["relevant_docs"])
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        print(
            f"- Query: {testcase['query']}"
            f"\n\t- Precision@{limit}: {precision:.4f}"
            f"\n\t- Recall@{limit}: {recall:.4f}"
            f"\n\t- F1 Score: {f1:.4f}"
            f"\n\t- Retrieved: {', '.join(retrieved)}"
            f"\n\t- Relevant: {', '.join(testcase['relevant_docs'])}"
        )


if __name__ == "__main__":
    main()
