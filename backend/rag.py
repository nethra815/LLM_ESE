"""Two-stage retrieval over the disease knowledge base."""
import numpy as np
from docx import Document
from sentence_transformers import CrossEncoder, SentenceTransformer


class DiseaseRAG:
    def __init__(
        self,
        docx_path: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_k: int = 3,
    ):
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")
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
        self.reranker = CrossEncoder(rerank_model)
        self.retrieval_k = min(retrieval_k, len(self.paras))

    def retrieve(self, query: str):
        """Retrieve three dense candidates, then return the cross-encoder winner."""
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.vec @ q
        candidate_indices = np.argsort(dense_scores)[-self.retrieval_k:][::-1]
        pairs = [(query, self.paras[i]) for i in candidate_indices]
        rerank_scores = np.asarray(self.reranker.predict(pairs)).reshape(-1)
        best_candidate = int(np.argmax(rerank_scores))
        i = int(candidate_indices[best_candidate])
        return self.names[i], self.paras[i], float(dense_scores[i])

    def diseases(self):
        return list(self.names)
