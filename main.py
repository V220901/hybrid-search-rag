"""Command-line entry point.

Usage:
    python main.py                                   # run the demo questions
    python main.py "How much vacation do I get?" --company Acme-India
    python main.py --docs-dir my_docs/ "What is the refund policy?"
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hybrid_rag.config import Config  # noqa: E402
from hybrid_rag.pipeline import RAGPipeline  # noqa: E402

DEMO_QUESTIONS = [
    ("How much vacation do I get?", "Acme-India"),
    ("How much vacation do I get?", "Acme-US"),
    ("I'm getting error E1047, what should I do?", None),
    ("What is the dress code on Fridays?", None),   # not in the docs -> should refuse
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search RAG (BM25 + dense + RRF + cross-encoder)")
    parser.add_argument("question", nargs="?", help="question to ask (omit to run the demo questions)")
    parser.add_argument("--company", choices=["Acme-India", "Acme-US"], help="filter policies by company")
    parser.add_argument("--docs-dir", help="folder of .md files to use instead of the sample Acme docs")
    parser.add_argument("--top-n", type=int, default=Config.top_n, help="chunks passed to the LLM")
    parser.add_argument("--model", default=Config.llm_model, help="LLM used on GPU")
    args = parser.parse_args()

    config = dataclasses.replace(Config(), top_n=args.top_n, llm_model=args.model)
    pipeline = RAGPipeline(config, docs_dir=args.docs_dir)
    print(f"Indexed {len(pipeline.chunks)} chunks | device={pipeline.device} | LLM={pipeline.llm_name}\n")

    questions = [(args.question, args.company)] if args.question else DEMO_QUESTIONS

    line = "=" * 80
    for question, company in questions:
        result = pipeline.ask(question, company)
        print(f"{line}\nQUESTION: {result.question}" + (f"  (company: {company})" if company else ""))
        print(f"{line}\nANSWER:\n{result.answer}\n")
        print("EVIDENCE (after rerank):")
        for n, chunk in enumerate(result.evidence, start=1):
            print(f"[{n}] {chunk['meta']['source']} › {chunk['meta']['section']}")
        print()


if __name__ == "__main__":
    main()
