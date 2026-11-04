# Disease Q&A Chatbot (Few-Shot Fine-tuning + RAG)

A chatbot that answers questions about 100 synthetic diseases in a fixed five-section format
(Disease, Symptoms, Causes, Prescription, Treatment).

- **Retrieval:** ChromaDB stores `all-MiniLM-L6-v2` embeddings over `data/disease_knowledge_base_complex.docx`, retrieves the top 3 disease candidates, then `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks them before generation.
- **Generation:** Qwen2.5-0.5B-Instruct with a LoRA adapter fine-tuned on 10 diseases (80 examples).
- **Frontend:** a plain HTML/JS chat page served by the same FastAPI app.

> Synthetic educational data. Prescriptions are generic drug classes without doses. **Not medical advice.**

## Repository layout

```
backend/        FastAPI app, retrieval (rag.py), generator (model.py)
frontend/       chat UI (index.html, style.css, app.js)
data/           knowledge base (100 diseases, Word document)
adapters/       place the trained LoRA adapter here (see adapters/README.md)
notebooks/      training and evaluation notebook (Colab)
results/        evaluation CSVs
```

## Run locally

```bash
git clone <YOUR-GITHUB-REPO-URL>
cd disease-rag-chatbot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy the adapter folder into adapters/disease-lora-fewshot/
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8000. The first start downloads the base model, embedding model, and cross-encoder reranker. ChromaDB persists its local index in `chroma/`.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Base LLM from Hugging Face |
| `ADAPTER_DIR` | `adapters/disease-lora-fewshot` | Path to the LoRA adapter |
| `KB_PATH` | `data/disease_knowledge_base_complex.docx` | Knowledge base document |
| `MIN_SCORE` | `0.35` | Below this similarity the bot says the condition is unknown |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to rerank the top 3 dense candidates |
| `CHROMA_DIR` | `chroma` | Local directory for the persistent ChromaDB index |
| `CHROMA_COLLECTION` | `disease_knowledge` | ChromaDB collection name |

## API

- `GET /api/diseases` returns the list of disease names.
- `POST /api/chat` with `{"message": "What causes Asthma?"}` returns `{answer, disease, score}`.

## Deploy

**Docker (any container host, e.g. a VPS, Render, Hugging Face Spaces with Docker SDK):**

```bash
docker build -t disease-chatbot .
docker run -p 7860:7860 disease-chatbot
```

The adapter must be present in `adapters/` before building. The container listens on port 7860.

Public deployment URL: **<INSERT DEPLOYMENT URL>**

## Evaluation

`notebooks/training_and_evaluation.ipynb` reproduces training and all metrics.
`results/ft_rag_all100.csv` holds the fine-tuned + RAG answers for all 100 diseases.
`results/sample_results.csv` holds the four-way comparison (Base, Fine-tuned, Base + RAG, Fine-tuned + RAG).

## Known limitations

- Scores come from a synthetic, uniform-style knowledge base. They do not show clinical accuracy.
- The `MIN_SCORE` threshold is not validated on real user questions.
- ChromaDB retrieval currently sends the top 3 candidates to the cross-encoder for reranking.
- Three of 100 answers omitted the "Treatment" header in the reported run.
- The model is small (0.5B parameters) and may produce fluent but unsupported text when context is missing.
