"""Two-stage retrieval over the disease knowledge base."""
import numpy as np
import chromadb
from docx import Document
from sentence_transformers import CrossEncoder, SentenceTransformer


class DiseaseRAG:
    def __init__(
        self,
        docx_path: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_k: int = 3,
        chroma_dir: str = "chroma",
        collection_name: str = "disease_knowledge",
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
        vectors = self.emb.encode(self.paras, normalize_embeddings=True)
        self.chroma = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.upsert(
            ids=[f"disease-{i}" for i in range(len(self.paras))],
            embeddings=vectors.tolist(),
            documents=self.paras,
            metadatas=[{"name": name} for name in self.names],
        )
        self.reranker = CrossEncoder(rerank_model)
        self.retrieval_k = min(retrieval_k, len(self.paras))

    def retrieve(self, query: str):
        """Retrieve three dense candidates, then return the cross-encoder winner."""
        q = self.emb.encode([query], normalize_embeddings=True)[0]
        result = self.collection.query(
            query_embeddings=[q.tolist()],
            n_results=self.retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        pairs = [(query, document) for document in documents]
        rerank_scores = np.asarray(self.reranker.predict(pairs)).reshape(-1)
        best_candidate = int(np.argmax(rerank_scores))
        best_distance = distances[best_candidate]
        return (
            metadatas[best_candidate]["name"],
            documents[best_candidate],
            float(1 - best_distance),
        )

    def diseases(self):
        return list(self.names)
