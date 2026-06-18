import re


def semantic_chunk(text, size, overlap):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]
    chunks = []
    i = 0
    while i < len(sentences):
        batch = [s.strip() for s in sentences[i : i + size]]
        batch = [s for s in batch if s]
        if batch:
            chunks.append(" ".join(batch))
        i += size - overlap
    return chunks
