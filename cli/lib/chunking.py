import re


def semantic_chunk(text, size, overlap):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    i = 0
    while i < len(sentences):
        batch = sentences[i:i+size]
        chunks.append(' '.join(batch))
        i += size - overlap
    return chunks
