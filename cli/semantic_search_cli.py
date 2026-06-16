#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.utils import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser(
        "verify", help="Prints the semantic model and its sequence length"
    )
    subparser.add_parser(
        "verify_embeddings", help="Transform our movies into a searchable vector database and verify"
    )
    embed_text_parser=subparser.add_parser(
        "embed_text",
        help="Generate the text embeddings"
    )
    embed_text_parser.add_argument("text",type=str,help="Text to generate the Embeddings")
    
    embed_query_parser=subparser.add_parser(
        "embed_query",
        help="Generate the text embeddings"
    )
    embed_query_parser.add_argument("query",type=str,help="Query to generate the Embeddings")
    
    search_parser = subparser.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Top N documents"
    )
    
    chunk_parser = subparser.add_parser(
        "chunk", help="Split the text into chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to divive into chunks")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words in each chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=2, help="Number of words to overlap in each chunk"
    )
    
    semantic_chunk_parser = subparser.add_parser(
        "semantic_chunk", help="Split the text into semantic chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to divive into chunks")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Number of semantics in each chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of words to overlap in each semantic chunk"
    )
    
    subparser.add_parser(
        "embed_chunks", help="Load movies documents and build the chunk embeddings"
    )
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_movies(args.query,args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks=chunk_text(args.text,args.chunk_size,args.overlap)
            for i,chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
            
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            semantic_chunks=semantic_chunk(args.text,args.max_chunk_size,args.overlap)
            for i,chunk in enumerate(semantic_chunks):
                print(f"{i+1}. {chunk}")
        
        case "embed_chunks":
            embed_chunks()
                
            
        case _:
            parser.print_help()



    


if __name__ == "__main__":
    main()
