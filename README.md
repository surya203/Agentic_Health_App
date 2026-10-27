# Cancer Awareness Assistant

Local engineering prototype. It searches PDF, DOCX, and TXT files with a local embedding model and Chroma, then asks Groq to draft an answer only when Python safety and evidence checks pass.

This is not a diagnostic system and it is not a clinically validated product. The files in `documents/` are synthetic samples. They are not hospital-approved clinical content.

## What it does

- Answers from retrieved passages and shows source citations.
- Refuses when the indexed documents do not support an answer, without asking Groq to invent one.
- Refuses to diagnose cancer, rule cancer out, or recommend treatment.
- Returns a fixed urgent-care message for selected emergency phrases before retrieval.

Questions and matching passages are sent to Groq only after those checks pass. Do not type personal or identifiable health information.

## Setup (Windows)

Python 3.12 is used for this prototype because the embedding libraries publish wheels for it. Python 3.10 or newer is the general requirement; use an interpreter those libraries support.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
$env:PIP_CONFIG_FILE = "nul"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`PIP_CONFIG_FILE=nul` skips machine-wide pip settings. Use it if install stalls on an extra package index. The CPU PyTorch wheel is enough for local embeddings.

Create a `.env` file in this folder. Get a key from the [Groq console](https://console.groq.com/home).

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` can be changed to another chat model enabled on the account. The app reads `Groq_api_key` if `GROQ_API_KEY` is unset. The key is loaded from `.env` and is not written into source or logs.

## Run

1. Put PDF, DOCX, or TXT files in `documents/`. Keep identifiable patient records out of this prototype.
2. Optional header lines read into chunk metadata:

```text
APPROVAL_STATUS: sample-not-clinically-approved
DOCUMENT_VERSION: 1.0
```

3. Build the local index:

```powershell
python ingest.py
```

4. Start the interface:

```powershell
streamlit run app.py
```

Re-run `python ingest.py` after documents change. That rebuilds `chroma_db/`. `python build_samples.py` rewrites the sample DOCX and PDF. `documents/warning_signs.txt` is stored as text.

## Sample checks

- A question the samples support, such as persistent hoarseness or who chooses a screening test, should cite the sample filename.
- A question absent from the samples, such as a clinic parking permit, should be refused without a generated medical answer.
- "Do I have cancer?" should be refused.
- A message about chest pain and trouble breathing should return the urgent-care message.

## Limits

Similarity scores are not medical confidence. A language model can still write an unsupported sentence, so citation checks can replace an answer with a refusal. The urgent-phrase list is not an emergency detector and will miss many emergencies. Review privacy, security, and clinical governance before any real patient use.

Groq API docs: https://console.groq.com/docs/openai
