"""LLM loading and generation (Hugging Face transformers)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, model_id: str, device: str, max_new_tokens: int = 200):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def make_message(system_content: str, user_content: str) -> list[dict]:
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @torch.no_grad()
    def generate(self, system_content: str, user_content: str) -> str:
        messages = self.make_message(system_content, user_content)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,                       # deterministic answers
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
