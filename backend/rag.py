"""Retrieval over the 100-disease knowledge base (one paragraph per disease)."""
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer


class DiseaseRAG:
    def __init__(self, docx_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.names, self.paras = [], []
        current = None
        # "Heading 2" = "N. Disease name"; the next non-empty paragraph is its text.
        for p in Document(docx_path).paragraphs:
            if p.style.name.startswith("Heading 2"):
                current = p.text.split(". ", 1)[1].strip()
            elif current and p.text.strip():
                self.names.append(current)
                self.paras.append(p.text.strip())
                current = None
        self.emb = SentenceTransformer(embed_model)
        self.vec = self.emb.encode(self.paras, normalize_embeddings=True)

    def retrieve(self, query: str):
        """Return (disease name, paragraph, cosine similarity) for the best match."""
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        scores = self.vec @ q
        i = int(np.argmax(scores))
        return self.names[i], self.paras[i], float(scores[i])

    def diseases(self):
        return list(self.names)
