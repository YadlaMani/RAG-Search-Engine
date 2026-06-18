import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.hybrid_search_lib.utils import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalizes the given values in the range 0-1"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="Scores to normalize"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search",
        help="Search movies using both keyword and semantic search with added weights",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="alpha is just a constant that we can use to dynamically control the weighting between the two scores",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Top N documents"
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search",
        help="Search movies using both keyword and semantic search with rrf",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        nargs="?",
        default=60,
        help="The k parameter (a constant) controls how much more weight we give to higher-ranked results vs. lower-ranked ones",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Top N documents"
    )

    args = parser.parse_args()
    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
