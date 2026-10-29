"""Retrieve local documents first, then search the web or answer generally."""

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
WHISPER_MODEL = "whisper-large-v3-turbo"

URGENT_MESSAGE = (
    "This may need urgent medical care. Contact emergency services or a clinician now. "
    "This tool cannot assess emergencies."
)
SCOPE_MESSAGE = (
    "I can't diagnose cancer, rule it out, or recommend a personal treatment. "
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

# Database column keys, with phrases a person might use for each one.
PROFILE_FIELDS = (
    ("name", "Hospital name", "The name of the hospital.", ("hospital name", "name")),
    ("address", "Address", "The street address of the hospital.", ("address", "location")),
    ("phone", "Phone number", "The phone number used to call the hospital.", ("phone number", "phone", "telephone")),
    ("email", "Email", "The email address for the hospital.", ("email", "e-mail")),
    ("website", "Website", "The hospital website address.", ("website", "web address")),
    ("logo_url", "Logo", "The web address of the hospital logo.", ("logo url", "logo")),
    ("colors", "Colors", "The hospital brand colors.", ("colors", "color", "colours", "colour")),
    ("fonts", "Fonts", "The typefaces used by the hospital.", ("fonts", "font", "typeface")),
    ("font_sizes", "Font sizes", "The text sizes used by the hospital.", ("font sizes", "font size", "text size")),
    ("font_weights", "Font weights", "How thick the hospital text is, such as normal or bold.", ("font weights", "font weight")),
    ("font_styles", "Font styles", "The text style, such as normal or italic.", ("font styles", "font style")),
    ("font_variants", "Font variants", "Extra text variants, such as small caps.", ("font variants", "font variant", "small caps", "small-caps")),
)

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

FALLBACK_PROMPT = (
    "You are a cancer-awareness education assistant. "
    "The hospital document files do not contain an answer to this question. "
    "Use browser search when the question needs current or outside information. "
    "If a web search is not needed, answer from general knowledge. "
    "Always give a useful educational answer. "
    "Do not diagnose a person, rule cancer out for a person, or give a personal treatment plan or dose. "
    "Begin with this sentence: This answer is not from the hospital documents. "
    "When you use web pages, end with a Sources section that lists each page title and URL. "
    "When you do not use the web, end with: Sources: general knowledge, not the hospital documents."
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
    r"\b(?:recommend|start|prescribe)(?: a)? (?:treatment|chemo(?:therapy)?|medication|medicine)\b",
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
# Filenames may contain spaces and parentheses, as in "Artificial Intelligence.pdf".
FILENAME_RE = (
    r"[A-Za-z0-9][A-Za-z0-9_\-]*(?:[ ()][A-Za-z0-9][A-Za-z0-9_\-]*)*"
    r"\.(?:txt|docx|pdf)"
)
CITATION_RE = re.compile(
    rf"({FILENAME_RE})\s*\(page\s*(n/a|\d+)\)",
    re.IGNORECASE,
)
FILE_RE = re.compile(FILENAME_RE, re.IGNORECASE)
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


def _fallback_complete(
    client: OpenAI,
    model: str,
    question: str,
    temperature: float,
    use_search: bool,
):
    request = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": FALLBACK_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    if use_search:
        request["tools"] = [{"type": "browser_search"}]
        request["tool_choice"] = "auto"
    return client.chat.completions.create(**request)


def _message_text(response) -> str:
    message = response.choices[0].message
    return (message.content or "").strip()


def _groq_client() -> tuple[OpenAI | None, str]:
    api_key, model = load_settings()
    if not api_key:
        return None, model
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=60.0, max_retries=0)
    return client, model


def _audio_upload(audio: bytes, filename: str) -> tuple[str, bytes, str]:
    if audio[:4] == b"RIFF":
        return "speech.wav", audio, "audio/wav"
    if audio[:4] == b"OggS":
        return "speech.ogg", audio, "audio/ogg"
    if filename.lower().endswith(".webm") or audio[:4] == b"\x1aE\xdf\xa3":
        return "speech.webm", audio, "audio/webm"
    return "speech.wav", audio, "audio/wav"


def transcribe_speech(audio: bytes, filename: str = "speech.wav") -> str:
    """Turn a microphone recording into the question text."""
    client, _model = _groq_client()
    if client is None or not audio:
        return ""
    name, payload, media_type = _audio_upload(audio, filename)
    transcript = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(name, payload, media_type),
        language="en",
    )
    text = getattr(transcript, "text", transcript)
    return str(text or "").strip()


def generate_from_evidence(question: str, context: str) -> str:
    client, model = _groq_client()
    if client is None:
        return TECHNICAL_MESSAGE
    try:
        try:
            response = _complete(client, model, question, context, 0.1)
        except APIError as exc:
            if "temperature" not in str(exc).lower():
                raise
            response = _complete(client, model, question, context, 1)
        return _message_text(response)
    except Exception:
        return TECHNICAL_MESSAGE


def present_outside_answer(text: str) -> str:
    cleaned = re.sub(r"【[^】]*】", "", text).strip()
    if "not from the hospital documents" not in cleaned.lower():
        cleaned = "This answer is not from the hospital documents.\n\n" + cleaned
    if re.search(r"https?://", cleaned, re.IGNORECASE):
        cleaned = re.sub(
            r"(?im)^sources:\s*general knowledge, not the hospital documents\.?\s*$",
            "",
            cleaned,
        ).strip()
    return cleaned


def generate_outside_documents(question: str) -> str:
    """Answer when the files have no supporting passage. Search the web only if needed."""
    client, model = _groq_client()
    if client is None:
        return TECHNICAL_MESSAGE
    attempts = (
        (True, 0.2),
        (True, 1),
        (False, 0.2),
    )
    last_error = ""
    for use_search, temperature in attempts:
        try:
            response = _fallback_complete(client, model, question, temperature, use_search)
            text = _message_text(response)
            if not text:
                continue
            return present_outside_answer(text)
        except Exception as exc:
            last_error = str(exc).lower()
            if (
                use_search
                and "temperature" not in last_error
                and "tool" not in last_error
                and "browser" not in last_error
            ):
                break
    return TECHNICAL_MESSAGE


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
    mentioned = set()
    for match in FILE_RE.finditer(cleaned):
        found = match.group(0).lower()
        hits = [name for name in allowed_names if found == name or found.endswith(name)]
        mentioned.add(hits[0] if len(hits) == 1 else found)
    if not mentioned <= allowed_names:
        return None
    citations = []
    for name, page in CITATION_RE.findall(cleaned):
        found = name.lower()
        hits = [source for source in allowed_names if found == source or found.endswith(source)]
        citations.append((hits[0] if len(hits) == 1 else found, _parse_page(page)))
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


def _format_field_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if value is None:
        return "not set"
    return str(value)


def current_hospital_profile() -> dict:
    """Prefer the live Supabase row, then the local profile values."""
    try:
        from supabase_store import HOSPITAL_PROFILE, configured, load_hospital_profile

        if configured():
            return load_hospital_profile()
        return dict(HOSPITAL_PROFILE)
    except Exception:
        from supabase_store import HOSPITAL_PROFILE

        return dict(HOSPITAL_PROFILE)


def matching_profile_fields(question: str) -> list[tuple]:
    """Map a question onto hospital profile keys by meaning, not exact file text."""
    lowered = question.lower()
    matched = []
    for field in PROFILE_FIELDS:
        aliases = sorted(field[3], key=len, reverse=True)
        for alias in aliases:
            if alias == "name" and "hospital" not in lowered and "key" not in lowered:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                matched.append(field)
                break
    if matched:
        return matched
    asks_about_keys = re.search(r"\b(keys?|fields?|columns?)\b", lowered) and re.search(
        r"\b(mean|meaning|explain|list|what|define|definition|describe|show)\b",
        lowered,
    )
    if asks_about_keys:
        return list(PROFILE_FIELDS)
    return []


def answer_from_profile(question: str) -> str | None:
    fields = matching_profile_fields(question)
    if not fields:
        return None
    profile = current_hospital_profile()
    lines = ["From the hospital profile stored in the database:"]
    for key, label, meaning, _aliases in fields:
        value = _format_field_value(profile.get(key))
        lines.append(f"{label} ({key}): {value}. {meaning}")
    lines.append("")
    lines.append("Sources: Supabase hospital_profile")
    return "\n".join(lines)


def answer_from_passages(passages: list[dict]) -> str:
    chunks = []
    labels = []
    for passage in passages[:3]:
        text = (passage.get("text") or "").strip()
        if text:
            chunks.append(text)
        label = citation_label(passage.get("metadata") or {})
        if label not in labels:
            labels.append(label)
    body = "\n\n".join(chunks)
    return "From the hospital documents:\n\n" + body + "\n\nSources: " + "; ".join(labels)


def answer_question(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return EMPTY_MESSAGE
    if _matches(text, URGENT_PATTERNS):
        return URGENT_MESSAGE
    if _matches(text, SCOPE_PATTERNS):
        return SCOPE_MESSAGE
    profile_answer = answer_from_profile(text)
    if profile_answer:
        return profile_answer
    try:
        passages = retrieve(text)
    except Exception:
        passages = []
    supported = [
        passage
        for passage in passages
        if passage["similarity"] >= SIMILARITY_THRESHOLD
    ]
    if supported:
        context = format_context(supported)
        generated = generate_from_evidence(text, context)
        validated = None if generated == TECHNICAL_MESSAGE else validate_answer(
            generated, supported, context
        )
        if validated:
            return validated
        return answer_from_passages(supported)
    return generate_outside_documents(text)
