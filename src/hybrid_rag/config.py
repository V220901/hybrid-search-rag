"""Configuration for the hybrid search RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # bi-encoder, 384-dim
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"     # cross-encoder reranker
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"                 # falls back to 0.5B on CPU
    cpu_llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    candidates_k: int = 8      # shortlist size from hybrid search
    top_n: int = 3             # chunks passed to the LLM after reranking
    rrf_k: int = 60            # Reciprocal Rank Fusion constant
    max_new_tokens: int = 200


RAG_SYSTEM_PROMPT = (
    "Answer using ONLY the numbered context. Cite passages like [1]. "
    "If the answer is not in the context, say: I don't know based on the Acme documents."
)
