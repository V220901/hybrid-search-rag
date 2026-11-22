"""Hybrid search RAG: BM25 + dense retrieval, RRF fusion, cross-encoder reranking and a local LLM."""

from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
