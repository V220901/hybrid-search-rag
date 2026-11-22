"""Sparse (BM25), dense (FAISS) and hybrid retrieval, plus cross-encoder reranking."""

import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def tokenize(text: str) -> list[str]:
    """Lowercase and keep only alphanumeric tokens (drops punctuation)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Search:
    """Keyword search: strong on exact terms such as error codes and IDs."""

    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k]]


class DenseSearch:
    """Semantic search with normalised embeddings in an exact inner-product FAISS index (= cosine)."""

    def __init__(self, texts: list[str], model_name: str, device: str):
        self.embedder = SentenceTransformer(model_name, device=device)
        vecs = self.embedder.encode(texts, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int) -> list[int]:
        qv = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(qv, min(k, self.index.ntotal))
        return [int(i) for i in ids[0]]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(doc) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


class HybridRetriever:
    """BM25 + dense retrieval fused with RRF."""

    def __init__(self, texts: list[str], embed_model: str, device: str, rrf_k: int = 60):
        self.bm25 = BM25Search(texts)
        self.dense = DenseSearch(texts, embed_model, device)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[int]:
        rankings = [self.bm25.search(query, k), self.dense.search(query, k)]
        return rrf(rankings, self.rrf_k)[:k]


class Reranker:
    """Cross-encoder that reads (question, chunk) together: slower but more precise, so used on a shortlist only."""

    def __init__(self, model_name: str, device: str):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: list[str], candidate_ids: list[int], top_n: int) -> list[int]:
        if not candidate_ids:
            return []
        scores = self.model.predict([(query, texts[i]) for i in candidate_ids])
        return [candidate_ids[j] for j in np.argsort(-scores)[:top_n]]
