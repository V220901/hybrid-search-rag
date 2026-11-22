"""End-to-end RAG pipeline: chunk -> hybrid retrieve -> filter -> rerank -> augment -> generate."""

from dataclasses import dataclass

import torch

from .config import RAG_SYSTEM_PROMPT, Config
from .data import ALL_COMPANIES, chunk_documents, load_documents
from .generation import Generator
from .retrieval import HybridRetriever, Reranker


@dataclass
class RAGResult:
    question: str
    company: str | None
    answer: str
    evidence: list[dict]          # reranked chunks, in the order they were numbered for the LLM


class RAGPipeline:
    def __init__(self, config: Config | None = None, docs_dir: str | None = None, load_llm: bool = True):
        self.config = config or Config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.chunks = chunk_documents(load_documents(docs_dir))
        self.texts = [c["text"] for c in self.chunks]

        self.retriever = HybridRetriever(self.texts, self.config.embed_model, self.device, self.config.rrf_k)
        self.reranker = Reranker(self.config.rerank_model, self.device)

        llm = self.config.llm_model if self.device == "cuda" else self.config.cpu_llm_model
        self.llm_name = llm
        self.generator = Generator(llm, self.device, self.config.max_new_tokens) if load_llm else None

    def retrieve(self, question: str, company: str | None = None) -> list[dict]:
        """Hybrid search a shortlist, keep chunks for the user's company, then rerank."""
        candidates = self.retriever.search(question, self.config.candidates_k)
        if company:
            candidates = [i for i in candidates
                          if self.chunks[i]["meta"]["company"] in (company, ALL_COMPANIES)]
        top = self.reranker.rerank(question, self.texts, candidates, self.config.top_n)
        return [self.chunks[i] for i in top]

    @staticmethod
    def build_prompt(evidence: list[dict], question: str) -> str:
        context = "\n".join(f"[{n}] {c['text']}" for n, c in enumerate(evidence, start=1))
        return f"Context:\n{context}\n\nQuestion: {question}"

    def ask(self, question: str, company: str | None = None) -> RAGResult:
        if self.generator is None:
            raise RuntimeError("Pipeline was created with load_llm=False; use retrieve() instead.")
        evidence = self.retrieve(question, company)
        answer = self.generator.generate(RAG_SYSTEM_PROMPT, self.build_prompt(evidence, question))
        return RAGResult(question, company, answer, evidence)
