import string
from nltk.stem import PorterStemmer


class Preprocessor:
    def __init__(self, stopwords_file="./data/stopwords.txt"):
        self.stemmer = PorterStemmer()
        self.translator = str.maketrans("", "", string.punctuation)

        with open(stopwords_file, "r") as f:
            self.stopwords = {line.strip().lower() for line in f if line.strip()}

    def normalize(self, text: str) -> list[str]:
        """
        Lowercase + remove punctuation + split
        """
        return text.lower().translate(self.translator).split()

    def remove_stopwords(self, words: list[str]) -> list[str]:
        return [word for word in words if word not in self.stopwords]

    def stem(self, words: list[str]) -> list[str]:
        return [self.stemmer.stem(word) for word in words]

    def process(self, text: str) -> list[str]:
        words = self.normalize(text)
        words = self.remove_stopwords(words)
        words = self.stem(words)
        return words
