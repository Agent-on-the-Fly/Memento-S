
from typing import List

from core.config.logging import get_logger

logger = get_logger(__name__)


class Qwen3EmbeddingFunction:

    QUERY_INSTRUCTION = (
        "Instruct: Given a user query, retrieve relevant skill descriptions "
        "that match the query\nQuery:"
    )

    @staticmethod
    def name() -> str:
        return "qwen3_embedding"

    def get_config(self) -> dict:
        return {
            "tokenizer_path": getattr(self, "_tokenizer_path", ""),
            "model_path": getattr(self, "_model_path", ""),
        }

    @staticmethod
    def build_from_config(config: dict) -> "Qwen3EmbeddingFunction":
        return Qwen3EmbeddingFunction(
            tokenizer_path=config.get("tokenizer_path", ""),
            model_path=config.get("model_path", ""),
        )

    def __init__(
        self,
        tokenizer_path: str,
        model_path: str,
        max_length: int = 8192,
        batch_size: int = 64,
        device: str = "auto",
    ):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer_path = tokenizer_path
        self._model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch

        logger.info("Loading Qwen3 embedding: tokenizer=%s, model=%s", tokenizer_path, model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

        if device == "auto":
            if torch.cuda.is_available():
                target_device = "cuda"
                target_dtype = torch.bfloat16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                target_device = "mps"
                target_dtype = torch.float16
            else:
                target_device = "cpu"
                target_dtype = torch.float32
        else:
            target_device = device
            target_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

        logger.info("Target device=%s, dtype=%s", target_device, target_dtype)

        if target_device == "cuda":
            try:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=target_dtype,
                    device_map="auto",
                )
                logger.info("Qwen3 model loaded with flash_attention_2")
            except Exception as e:
                logger.info("flash_attention_2 not available, using default: %s", e)
                self.model = AutoModel.from_pretrained(
                    model_path, torch_dtype=target_dtype,
                ).to(target_device)
        else:
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=target_dtype,
            ).to(target_device)

        self.model.eval()
        self._device = target_device

        with torch.no_grad():
            dummy = self.tokenizer(["test"], return_tensors="pt", padding=True).to(target_device)
            out = self.model(**dummy)
            self._dim = out.last_hidden_state.shape[-1]

        logger.info("Qwen3 embedding ready: dim=%d, device=%s", self._dim, target_device)

    @property
    def dimension(self) -> int:
        return self._dim

    def _last_token_pool(self, last_hidden_states, attention_mask):
        torch = self._torch
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _encode(self, texts: List[str]) -> List[List[float]]:
        torch = self._torch
        F = torch.nn.functional
        all_embeddings: List[List[float]] = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                outputs = self.model(**inputs)
                emb = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.extend(emb.cpu().float().tolist())

        return all_embeddings

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        return self._encode(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)
