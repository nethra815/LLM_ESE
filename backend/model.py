"""Fine-tuned generator: base Qwen2.5-0.5B-Instruct + LoRA adapter."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HEADER = "**Disease:**"


def make_user(question: str, context: str | None) -> str:
    # Must match the prompt format used in training.
    if context is None:
        return question
    return ("Context:\n" + context + "\n\nQuestion: " + question +
            "\nAnswer using only the context.")


def clean(answer: str) -> str:
    """Drop any preamble before the first header."""
    k = answer.find(HEADER)
    return answer[k:] if k >= 0 else answer


class Generator:
    def __init__(self, base_id: str, adapter_dir: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()

    @torch.no_grad()
    def answer(self, question: str, context: str) -> str:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": make_user(question, context)}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=300, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return clean(raw)
