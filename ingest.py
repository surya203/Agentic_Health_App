"""Parse approved sample documents, chunk them, and store embeddings in Chroma."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "cancer_awareness"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80
MAX_CHUNK_TOKENS = 480
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
DEFAULT_APPROVAL = "sample-not-clinically-approved"
DEFAULT_VERSION = "1.0"

HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,'/:\-]{2,80}$")


def read_txt(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.strip():
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_pdf(path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((index, text))
    return pages


def iter_docx_blocks(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def read_docx(path: Path) -> str:
    document = Document(str(path))
    lines: list[str] = []
    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                lines.append(text)
        else:
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rendered = " | ".join(cell for cell in cells if cell)
                if rendered:
                    lines.append(rendered)
    return "\n".join(lines).strip()


def load_document(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = read_txt(path)
        return [(0, text)] if text.strip() else []
    if suffix == ".docx":
        text = read_docx(path)
        return [(0, text)] if text.strip() else []
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"unsupported file type: {suffix}")


def parse_metadata(text: str) -> tuple[str, str]:
    approval = DEFAULT_APPROVAL
    version = DEFAULT_VERSION
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("APPROVAL_STATUS:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                approval = value
        elif stripped.startswith("DOCUMENT_VERSION:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                version = value
    return approval, version


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith("."):
        return False
    if "|" in stripped:
        return False
    if stripped.startswith("APPROVAL_STATUS:") or stripped.startswith("DOCUMENT_VERSION:"):
        return False
    if stripped.startswith("#"):
        return True
    return bool(HEADING_RE.match(stripped)) and 2 <= len(stripped.split()) <= 12


def split_sections(text: str) -> list[tuple[str, str]]:
    heading = ""
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        if is_heading(line):
            body = "\n".join(buffer).strip()
            if body:
                sections.append((heading, body))
            heading = line.strip().lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((heading, body))
    if not sections and text.strip():
        sections.append(("", text.strip()))
    return sections


def chunk_ids(token_ids: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    if not token_ids:
        return []
    step = max(1, chunk_size - overlap)
    windows: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        windows.append(token_ids[start : start + chunk_size])
        if start + chunk_size >= len(token_ids):
            break
        start += step
    return windows


def with_heading(heading: str, body: str, tokenizer) -> str:
    body = body.strip()
    if not body:
        return ""
    if heading and not body.startswith(heading):
        combined = f"{heading}\n{body}"
    else:
        combined = body
    token_ids = tokenizer.encode(combined, add_special_tokens=False)
    if len(token_ids) > MAX_CHUNK_TOKENS:
        token_ids = token_ids[:MAX_CHUNK_TOKENS]
        combined = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return combined


def chunk_page(text: str, tokenizer) -> list[str]:
    chunks: list[str] = []
    for heading, body in split_sections(text):
        token_ids = tokenizer.encode(body, add_special_tokens=False)
        for window in chunk_ids(token_ids, CHUNK_TOKENS, CHUNK_OVERLAP):
            piece = tokenizer.decode(window, skip_special_tokens=True).strip()
            combined = with_heading(heading, piece, tokenizer)
            if len(combined) >= 40:
                chunks.append(combined)
    return chunks


def create_collection(client: chromadb.PersistentClient):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    try:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )


def collect_records(tokenizer) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    failures: list[str] = []
    files = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and not path.name.startswith("~$")
    )
    for path in files:
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            print(f"Skipped unsupported file: {path.name}")
            continue
        try:
            pages = load_document(path)
        except Exception as exc:
            failures.append(path.name)
            print(f"Failed to parse {path.name}: {type(exc).__name__}")
            continue
        if not pages:
            print(f"Skipped empty file: {path.name}")
            continue
        chunk_index = 0
        file_chunks = 0
        for page_number, text in pages:
            approval, version = parse_metadata(text)
            for chunk in chunk_page(text, tokenizer):
                records.append(
                    {
                        "id": f"{path.name}::p{page_number}::c{chunk_index}",
                        "text": chunk,
                        "metadata": {
                            "source": path.name,
                            "page": int(page_number),
                            "approval_status": approval,
                            "document_version": version,
                        },
                    }
                )
                chunk_index += 1
                file_chunks += 1
        if file_chunks == 0:
            print(f"Skipped file with no usable text: {path.name}")
        else:
            print(f"Indexed {path.name}: {file_chunks} chunk(s)")
    return records, failures


def main() -> int:
    if not DOCUMENTS_DIR.is_dir():
        print("documents/ folder not found.")
        return 1
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    records, failures = collect_records(model.tokenizer)
    if not records:
        print("No chunks were indexed.")
        return 1 if failures else 0
    print(f"Embedding {len(records)} chunk(s)...")
    vectors = model.encode(
        [record["text"] for record in records],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = create_collection(client)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["text"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=vectors.tolist(),
    )
    print(f"Collection {COLLECTION_NAME} now has {collection.count()} chunk(s).")
    if failures:
        print("Files that could not be parsed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
