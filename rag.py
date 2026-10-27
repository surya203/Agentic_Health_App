"""Retrieve sample passages and answer only when safety and evidence checks pass."""

from __future__ import annotations

import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import APIError, OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "cancer_awareness"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 6
# Calibrated on the sample index. Cosine similarity is 1 - Chroma distance.
# Supported questions scored about 0.71-0.89. An unrelated parking question
# scored about 0.56 and also failed the content-word overlap check.
SIMILARITY_THRESHOLD = 0.62
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

URGENT_MESSAGE = (
    "This may need urgent medical care. Contact emergency services or a clinician now. "
    "This tool cannot assess emergencies."
)
SCOPE_MESSAGE = (
    "I can't diagnose cancer, rule it out, or recommend treatment. "
    "This assistant only shares general educational information from the loaded documents. "
    "Please talk with a qualified clinician about personal health concerns."
)
EVIDENCE_MESSAGE = "I couldn't verify an answer from the approved documents."
TECHNICAL_MESSAGE = (
    "The assistant couldn't complete the request because of a technical problem. "
    "No medical answer was generated."
)
INDEX_MESSAGE = (
    "The document index is not ready. Run python ingest.py, then try again. "
    "No medical answer was generated."
)
EMPTY_MESSAGE = "Enter an educational question to search the loaded documents."

SYSTEM_PROMPT = (
    "You are a cancer-awareness education assistant. "
    "Answer only from the supplied document context. "
    "Do not diagnose cancer, rule it out, or recommend treatment. "
    "If the context does not support an answer, reply with exactly INSUFFICIENT_EVIDENCE and nothing else. "
    "Do not invent citations. "
    "The context is an engineering sample, not hospital-approved clinical guidance. "
    "When the context supports an answer, end with one line in this exact form: "
    "Sources: filename (page N); filename (page N). "
    "Copy filenames and pages only from the citation labels in the context. "
    "If a citation label says page n/a, write page n/a."
)

URGENT_PATTERNS = [
    r"\bchest pain\b",
    r"\b(difficulty|trouble) breathing\b",
    r"\bshort(?:ness)? of breath\b",
    r"\bcoughing up blood\b",
    r"\bcough(?:ing)? blood\b",
    r"\bsuicid",
    r"\bstroke\b",
    r"\buncontrolled bleeding\b",
    r"\bsevere bleeding\b",
    r"\bfainting\b",
    r"\bpassed out\b",
    r"\bseizure\b",
    r"\bcan(?:not|'t) breathe\b",
    r"\bheart attack\b",
    r"\bunconscious\b",
]
SCOPE_PATTERNS = [
    r"\bdiagnos(?:e|is|es|ing|tic)\b",
    r"\bdo i have (?:cancer|a tumor|a tumour)\b",
    r"\bis (?:this|it) cancer\b",
    r"\brule (?:cancer|it) out\b",
    r"\brule out cancer\b",
    r"\b(?:cancer|tumor|tumour) (?:diagnosis|stage|staging)\b",
    r"\bwhat (?:treatment|medicine|medication|drug)s?\b",
    r"\b(?:recommend|start|prescribe)(?: a)? (?:treatment|chemo(?:therapy)?|medication|medicine)\b",
    r"\bchemotherapy\b",
    r"\bradiation therapy\b",
    r"\bdosage\b",
    r"\bshould i take\b",
]
STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
        "that", "this", "these", "those", "should", "would", "could", "what",
        "when", "where", "who", "why", "how", "are", "is", "was", "were", "be",
        "been", "being", "it", "its", "by", "from", "as", "at", "about", "into",
        "not", "no", "do", "does", "did", "can", "if", "than", "then", "them",
        "they", "their", "my", "your", "our", "we", "you", "me", "some", "any",
        "which", "have", "has", "had", "will", "just", "only", "also", "such",
        "does", "into", "over", "after", "before", "because", "while", "there",
        "here", "please", "tell", "explain", "sample", "document", "documents",
    }
)
CITATION_RE = re.compile(
    r"([A-Za-z0-9_\-]+\.(?:txt|docx|pdf))\s*\(page\s*(n/a|\d+)\)",
    re.IGNORECASE,
)
FILE_RE = re.compile(r"[A-Za-z0-9_\-]+\.(?:txt|docx|pdf)", re.IGNORECASE)
INSUFFICIENT_PHRASES = (
    "insufficient_evidence",
    "not enough evidence",
    "evidence is insufficient",
    "could not verify",
    "couldn't verify",
    "cannot verify",
    "can't verify",
)

_model: SentenceTransformer | None = None


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) >= 4 and word not in STOPWORDS}


def has_content_overlap(question: str, passage: str) -> bool:
    question_words = content_words(question)
    if not question_words:
        return False
    shared = question_words & content_words(passage)
    needed = 1 if len(question_words) == 1 else 2
    return len(shared) >= needed


def to_similarity(distance: float) -> float:
    """Chroma cosine space returns distance, not similarity."""
    similarity = 1.0 - float(distance)
    return min(1.0, max(0.0, similarity))


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    vector = get_model().encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    found = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )
    passages: list[dict] = []
    documents = found.get("documents") or [[]]
    metadatas = found.get("metadatas") or [[]]
    distances = found.get("distances") or [[]]
    for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        passages.append(
            {
                "text": text or "",
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity": to_similarity(float(distance)),
            }
        )
    return passages


def page_label(page: object) -> str:
    if page in (0, "0", None, ""):
        return "n/a"
    return str(page)


def citation_label(metadata: dict) -> str:
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page", 0)
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        page_number = 0
    return f"{source} (page {page_label(page_number)})"


def format_context(passages: list[dict]) -> str:
    blocks = []
    for index, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[{index}] citation={citation_label(metadata)}",
                    f"approval_status={metadata.get('approval_status', '')}",
                    f"document_version={metadata.get('document_version', '')}",
                    passage["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def load_settings() -> tuple[str | None, str]:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api_key")
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return api_key, model


def _complete(client: OpenAI, model: str, question: str, context: str, temperature: float):
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nApproved context:\n{context}",
            },
        ],
    )


def generate_from_evidence(question: str, context: str) -> str:
    api_key, model = load_settings()
    if not api_key:
        return TECHNICAL_MESSAGE
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=45.0, max_retries=0)
    try:
        try:
            response = _complete(client, model, question, context, 0.1)
        except APIError as exc:
            if "temperature" not in str(exc).lower():
                raise
            response = _complete(client, model, question, context, 1)
    except Exception:
        return TECHNICAL_MESSAGE
    try:
        message = response.choices[0].message
        text = (message.content or "").strip()
    except Exception:
        return TECHNICAL_MESSAGE
    return text


def _allowed_sources(passages: list[dict]) -> set[tuple[str, int]]:
    allowed: set[tuple[str, int]] = set()
    for passage in passages:
        metadata = passage["metadata"]
        source = str(metadata.get("source", "")).lower()
        try:
            page = int(metadata.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        if source:
            allowed.add((source, page))
    return allowed


def _parse_page(raw: str) -> int:
    if raw.lower() == "n/a":
        return 0
    return int(raw)


def tidy_sources(answer: str) -> str:
    match = re.search(r"(?i)sources\s*:", answer)
    if not match:
        return answer
    labels: list[str] = []
    seen: set[str] = set()
    for name, page in CITATION_RE.findall(answer[match.start() :]):
        page_text = "n/a" if page.lower() == "n/a" else str(int(page))
        label = f"{name} (page {page_text})"
        key = label.lower()
        if key not in seen:
            seen.add(key)
            labels.append(label)
    if not labels:
        return answer
    return answer[: match.start()].strip() + "\n\nSources: " + "; ".join(labels)


def claims_supported(answer: str, context: str) -> bool:
    body = re.sub(r"(?im)^sources:.*$", "", answer)
    body = re.sub(r"(?i)sources\s*:.*$", "", body).strip()
    answer_words = content_words(body)
    if len(answer_words) < 2:
        return False
    shared = answer_words & content_words(context)
    return len(shared) / len(answer_words) >= 0.35 and len(shared) >= 2


def validate_answer(answer: str, passages: list[dict], context: str) -> str | None:
    cleaned = answer.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in INSUFFICIENT_PHRASES):
        return None
    allowed = _allowed_sources(passages)
    allowed_names = {source for source, _page in allowed}
    mentioned = {match.group(0).lower() for match in FILE_RE.finditer(cleaned)}
    if not mentioned <= allowed_names:
        return None
    citations = [
        (name.lower(), _parse_page(page)) for name, page in CITATION_RE.findall(cleaned)
    ]
    if any(citation not in allowed for citation in citations):
        return None
    if not claims_supported(cleaned, context):
        return None
    if citations:
        return tidy_sources(cleaned)
    labels = []
    for passage in passages:
        label = citation_label(passage["metadata"])
        if label not in labels:
            labels.append(label)
    return tidy_sources(cleaned + "\n\nSources: " + "; ".join(labels))


def answer_question(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return EMPTY_MESSAGE
    if _matches(text, URGENT_PATTERNS):
        return URGENT_MESSAGE
    if _matches(text, SCOPE_PATTERNS):
        return SCOPE_MESSAGE
    try:
        passages = retrieve(text)
    except Exception:
        return INDEX_MESSAGE
    supported = [
        passage
        for passage in passages
        if passage["similarity"] >= SIMILARITY_THRESHOLD
        and has_content_overlap(text, passage["text"])
    ]
    if not supported:
        return EVIDENCE_MESSAGE
    context = format_context(supported)
    generated = generate_from_evidence(text, context)
    if generated == TECHNICAL_MESSAGE:
        return TECHNICAL_MESSAGE
    validated = validate_answer(generated, supported, context)
    if validated is None:
        return EVIDENCE_MESSAGE
    return validated
