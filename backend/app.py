"""FastAPI server: serves the chat frontend and the /api endpoints."""
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.rag import DiseaseRAG
from backend.model import Generator

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "adapters/disease-lora-fewshot")
KB_PATH = os.getenv("KB_PATH", "data/disease_knowledge_base_100.docx")
# Questions whose best paragraph scores below this are treated as out of scope.
# This threshold is a starting value and has not been validated; tune it on real questions.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Disease RAG Chatbot")
state = {}


class ChatIn(BaseModel):
    message: str


@app.on_event("startup")
def load():
    state["rag"] = DiseaseRAG(KB_PATH)
    state["gen"] = Generator(BASE_MODEL, ADAPTER_DIR, DEVICE)


@app.get("/api/diseases")
def diseases():
    return {"diseases": state["rag"].diseases()}


@app.post("/api/chat")
def chat(body: ChatIn):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")
    name, paragraph, score = state["rag"].retrieve(msg)
    if score < MIN_SCORE:
        return {"answer": "I could not find this condition in my knowledge base. "
                          "Try a disease name, for example 'Tell me about Asthma'.",
                "disease": None, "score": score}
    answer = state["gen"].answer(msg, paragraph)
    return {"answer": answer, "disease": name, "score": score}


# Mounted last so the /api routes take precedence.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
