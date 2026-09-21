"""Local Hugging Face causal-LM adapter; no top-k truncation or text generation."""
import math


class HuggingFaceBackend:
    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", device="auto", revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, trust_remote_code=False).to(device).eval()
        # Use the tokenizer's explicit end-of-turn/EOS token, including in scoring.
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Model tokenizer must define an EOS/end-of-turn token")
        limits = [getattr(self.model.config, "max_position_embeddings", None),
                  self.tokenizer.model_max_length]
        self.limit = min(x for x in limits if isinstance(x, int) and x > 0)

    def prompt_ids(self, system, form):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": form}],
                tokenize=True, add_generation_prompt=True)
        return self.tokenizer.encode(system + "\n\n" + form + "\n", add_special_tokens=True)

    def answer_ids(self, answer):
        ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if self.eos in ids or any(t in self.tokenizer.all_special_ids for t in ids):
            raise ValueError("Options may not contain special tokens")
        if self.tokenizer.decode(ids, skip_special_tokens=False) != answer:
            raise ValueError("Option does not round-trip through this tokenizer")
        return ids + [self.eos]

    def next_logprobs(self, prompt, prefix, allowed):
        ids = list(prompt) + list(prefix)
        if len(ids) > self.limit:
            raise ValueError(f"Input exceeds context limit ({self.limit}); no silent truncation")
        tensor = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=tensor, attention_mask=self.torch.ones_like(tensor),
                                use_cache=False).logits[0, -1].float()
            # Full vocabulary normalisation first, THEN select candidate tokens.
            selected = self.torch.log_softmax(logits, dim=-1)[list(allowed)].cpu().tolist()
        if any(not math.isfinite(x) for x in selected):
            raise ValueError("Model returned non-finite token probabilities")
        return dict(zip(allowed, selected))
