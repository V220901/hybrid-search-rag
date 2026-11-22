"""Knowledge base and header-based chunking."""

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

# Sample knowledge base for a fictional company, Acme Corp.
SAMPLE_DOCS = {
    "hr_policy_india.md": """# Acme-India HR Policy
## Annual leave
Employees accrue 18 days of paid leave annually. Unused leave of up to 30 days carries over into the next year.
## Sick leave
Employees get 12 days of sick leave per year. A medical certificate is needed for absences over 3 days.
## Remote work
Employees may work from home up to 2 days per week with their manager's approval.""",

    "hr_policy_us.md": """# Acme-US HR Policy
## Annual leave
Employees receive 15 days of paid time off (PTO) per year. PTO does not carry over.
## Sick leave
Employees get 10 days of sick leave per year.
## Remote work
Employees may work from home up to 3 days per week.""",

    "it_helpdesk.md": """# Acme IT Helpdesk FAQ
## Error E1047: login token expired
Appears when your single sign-on (SSO) session has expired. Fix: clear cookies for sso.acme.com and log in again.
## Error E2001: disk quota exceeded
Your network home drive has reached its 50 GB limit. Delete old files or request more space on the IT portal.
## Password reset
Passwords expire every 90 days. Reset yours at password.acme.com.""",

    "facilities.md": """# Acme Facilities Guide
## Cafeteria
The cafeteria is open from 8 am to 6 pm on working days. Lunch is subsidised.
## Gym
The office gym on the 3rd floor is open from 6 am to 10 pm.""",
}

# Which company each document applies to; anything else applies to everyone.
COMPANY_OF = {"hr_policy_india.md": "Acme-India", "hr_policy_us.md": "Acme-US"}
ALL_COMPANIES = "Acme (all)"


def load_documents(docs_dir: str | None = None) -> dict[str, str]:
    """Return ``{filename: markdown}``. Uses the built-in sample docs unless a folder of .md files is given."""
    if docs_dir is None:
        return dict(SAMPLE_DOCS)
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")
    return {f.name: f.read_text(encoding="utf-8") for f in files}


def chunk_documents(docs: dict[str, str]) -> list[dict]:
    """Split each document on its ``#`` / ``##`` headers into chunks with metadata.

    Each chunk is ``{"text": "<title> › <section>: <content>", "meta": {source, section, company}}``.
    """
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    chunks = []
    for source, text in docs.items():
        for piece in splitter.split_text(text):
            m = piece.metadata
            title, section = m.get("title", source), m.get("section", "")
            chunks.append({
                "text": f"{title} › {section}: {piece.page_content}",
                "meta": {"source": source, "section": section,
                         "company": COMPANY_OF.get(source, ALL_COMPANIES)},
            })
    return chunks
