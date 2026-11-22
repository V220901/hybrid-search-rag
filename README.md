# Hybrid Search RAG

A retrieval-augmented generation (RAG) assistant that answers employee questions over company policy documents (HR, IT helpdesk, facilities) for a fictional company, **Acme Corp**. It cites its sources and refuses when the documents don't cover the question.

Neither keyword search nor embeddings is enough on its own. **BM25** finds exact terms such as `E1047` but misses paraphrases ("vacation" vs "paid leave"). **Dense embeddings** catch the meaning but can blur exact identifiers. This project uses both, fuses them with **Reciprocal Rank Fusion**, reranks the shortlist with a **cross-encoder**, and filters by **metadata**, so an employee in India and one in the US each get answers from their own policy.

## How it works

```
Markdown docs ──> Header chunks (+ metadata) ──┬──> BM25 ─────────┐
                                               └──> MiniLM + FAISS ┴──> RRF ──> Company filter ──> Cross-encoder rerank ──> Numbered context ──> Qwen2.5 ──> Cited answer
```

| Stage | Module | What it does |
|---|---|---|
| Load + chunk | `src/hybrid_rag/data.py` | Splits markdown on `#` / `##` headers and tags each chunk with `source`, `section` and `company` |
| Sparse retrieval | `src/hybrid_rag/retrieval.py` | BM25 keyword ranking (`rank_bm25`) |
| Dense retrieval | `src/hybrid_rag/retrieval.py` | `all-MiniLM-L6-v2` embeddings, normalised, in a FAISS `IndexFlatIP` (inner product = cosine) |
| Fusion | `src/hybrid_rag/retrieval.py` | Reciprocal Rank Fusion of both rankings |
| Rerank | `src/hybrid_rag/retrieval.py` | `ms-marco-MiniLM-L6-v2` cross-encoder scores each (question, chunk) pair |
| Augment | `src/hybrid_rag/pipeline.py` | Metadata filter, then numbered context in a strict "answer only from context, cite [n]" prompt |
| Generate | `src/hybrid_rag/generation.py` | Qwen2.5-1.5B-Instruct (GPU) or 0.5B (CPU) via Hugging Face `transformers` |

## Project structure

```
hybrid-search-rag/
├── main.py                       # CLI entry point
├── requirements.txt
├── src/hybrid_rag/
│   ├── config.py                 # models, k values, system prompt
│   ├── data.py                   # sample knowledge base + header chunking
│   ├── retrieval.py              # BM25, FAISS dense search, RRF, cross-encoder rerank
│   ├── generation.py             # LLM wrapper
│   └── pipeline.py               # end-to-end orchestration
└── docs/
    └── sample_output.md          # example run
```

## Run it

**Requirements:** Python 3.10+. A **CUDA GPU** is recommended for the 1.5B model, and it also runs on CPU with the 0.5B model. No local GPU? Use Google Colab's free T4:

1. Open a new Colab notebook and set **Runtime → Change runtime type → T4 GPU**.
2. Run:

```python
!git clone https://github.com/V220901/hybrid-search-rag.git
%cd hybrid-search-rag
!pip install -q -r requirements.txt
!python main.py
```

`python main.py` with no question runs the four demo questions. To ask your own:

```bash
python main.py "How much vacation do I get?" --company Acme-India
python main.py "I'm getting error E1047, what should I do?"
python main.py --docs-dir my_docs/ "What is the refund policy?"   # use your own .md files
```

## Example

**Hybrid search fixes each method's blind spot:**

| Query | BM25 top-1 | Dense top-1 |
|---|---|---|
| `error E1047 when I log in` | Error E1047 ✅ | Error E1047 ✅ |
| `When does the canteen open?` | Gym ❌ | Cafeteria ✅ |

**End-to-end answers:**

| Question | Filter | Answer |
|---|---|---|
| How much vacation do I get? | Acme-India | 18 days of paid leave, cited ✅ |
| How much vacation do I get? | Acme-US | 15 days PTO, no carry-over, cited ✅ |
| What is the dress code on Fridays? | — | "I don't know based on the Acme documents." ✅ |
| I'm getting error E1047, what should I do? | — | Refused ⚠️ (see limitations) |

See [`docs/sample_output.md`](docs/sample_output.md) for the full output.

## Notes and limitations

- **E1047 miss:** retrieval and reranking put the correct chunk at rank 1, but the 1.5B model still refused. Generation was too conservative, not retrieval. A larger model or a less restrictive prompt should fix it.
- The knowledge base is small (11 chunks), so it shows the pipeline rather than testing it at scale. Point `--docs-dir` at a real document set to go further.
- There is no evaluation set yet. Next steps are measuring retrieval hit-rate@k and answer accuracy on a labelled question set, and comparing BM25, dense and hybrid retrieval.
- Metadata filtering here happens after retrieval. With a larger corpus, filter inside a vector DB (e.g. Chroma's `where=` filter) before searching, so the filter doesn't shrink the shortlist.

## Tech stack

Python · rank_bm25 · Sentence-Transformers (MiniLM bi-encoder + cross-encoder) · FAISS · LangChain text splitters · Hugging Face Transformers · PyTorch · Qwen2.5-Instruct
