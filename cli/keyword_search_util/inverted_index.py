import pickle
from collections import defaultdict, Counter
import json
from pathlib import Path
from cli.keyword_search_util.preprocessor import Preprocessor
import math
import cli.keyword_search_util.constants as constants
import os


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.processor = Preprocessor()
        self.doc_lengths = {}
        Path("cache").mkdir(exist_ok=True)
        self.doc_lengths_path = os.path.join(constants.CACHE_DIR, "doc_lengths.pkl")
        self.index_path = os.path.join(constants.CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(constants.CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            constants.CACHE_DIR, "term_frequencies.pkl"
        )

    def __add_document(self, doc_id, text):
        ptext = self.processor.process(text)
        self.doc_lengths[doc_id] = len(ptext)
        for token in ptext:
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        if len(self.doc_lengths) == 0:
            return 0.0
        return float(f"{total_length / len(self.doc_lengths):.2f}")

    def get_documents(self, term):
        return sorted(self.index[term])

    def build(self, movie_file="./data/movies.json"):
        with open(movie_file, "r") as f:
            data = json.load(f)
        movies = data["movies"]
        for doc_id, movie in enumerate(movies):
            self.docmap[doc_id + 1] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id + 1, text)

    def save(self):
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_bm25_tf(
        self, doc_id, term, k1=constants.BM25_K1, b=constants.BM25_B
    ) -> float:
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query, limit):
        processed_query = self.processor.process(query)
        scores = {}
        for term in processed_query:
            for doc_id in self.get_documents(term):
                if scores.__contains__(doc_id):
                    scores[doc_id] += self.bm25(doc_id, term)
                else:
                    scores[doc_id] = self.bm25(doc_id, term)

        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        res = {}
        for doc_id, score in sorted_scores.items():
            res[doc_id] = score
            if len(res) == limit:
                break
        return res
