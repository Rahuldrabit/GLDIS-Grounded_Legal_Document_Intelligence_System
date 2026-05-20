# Grounded Legal Document Intelligence System (GLDIS)

**Stack:** Python 3.11 · FastAPI · LM Studio OpenAI-compatible servers · FAISS · BM25 · Neo4j · Mem0  

---

## Overview

GLDIS is a modular legal document intelligence pipeline that ingests messy legal documents, extracts block-grounded text with real pixel coordinates, retrieves dense + sparse + graph-expanded evidence, and generates grounded legal drafts with a verifier loop and feedback memory.

```
Document
  │
  ├─[1] Stage 1: Ingest + Block Extraction ─ pdf2image · MinerU (optional) · InternVL2.5 INT4
  │                                          └ real pixel bbox [x1,y1,x2,y2] @ 300 DPI
  │
  ├─[2] Stage 2: Retrieval ─ FAISS (dense) + BM25 (sparse) + Neo4j GraphRAG 1–2 hop expansion
  │
  ├─[3] Stage 3: Reason + Verify ─ DeepSeek-R1-Distill-Llama-70B reasoner → Llama-4-8B verifier
  │                               └ bounded auto-correction loop + claim map + citations
  │
  └─[4] Stage 4: Memory + Learning ─ Mem0 feedback store + DPO export for offline preference tuning
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone <repo>
cd "grounded document intelligence system"
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
copy .env.example .env
# Edit .env — set LLM_PROVIDER=lmstudio, ollama, or openai
# Leave LLM_PROVIDER blank to keep the legacy auto-selection behavior
```

### 3. Run (backend + frontend)

Backend (development - FastAPI + Uvicorn):

```bash
# From repository root
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
# or (legacy) python main.py
# Open http://localhost:8000
```

Frontend (two options):

- Option A — Serve UI from the backend (no Node required)

```bash
# Start the backend (see commands above) and open http://localhost:8000
# The FastAPI server serves the zero-build UI from `static/` by default
```

- Option B — Frontend dev server (hot-reload, requires Node.js)

```bash
cd ui
npm install
npm run dev
# Vite dev server typically runs at http://localhost:5173
```

To build the frontend for production and preview the build locally:

```bash
cd ui
npm run build
npm run preview
# Or copy the built files into the backend `static/` directory to serve them from FastAPI
```

---

## Local Model Setup (Optional — for best accuracy)

1. Download [LM Studio](https://lmstudio.ai) and load the models you want to use for the pipeline:
  - VLM: `qwen/qwen3-vl-4b` or `qwen2.5-vl-*`
  - Reasoner: `deepseek-r1-distill-llama-70b` or a smaller local substitute for testing
  - Verifier: `llama-4-8b` or a smaller local substitute for testing
2. Start the LM Studio server and confirm it exposes an OpenAI-compatible API at `http://localhost:1234/v1`
3. Set in `.env` or your shell:
  ```
  VLM_ENABLED=true
  VLM_API_BASE=http://localhost:1234/v1
  VLM_MODEL=qwen/qwen3-vl-4b
  REASONING_API_BASE=http://localhost:1234/v1
  REASONING_MODEL=deepseek-r1-distill-llama-70b
  VERIFIER_API_BASE=http://localhost:1234/v1
  VERIFIER_MODEL=llama-4-8b
  NEO4J_ENABLED=true
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=gldis_secret
  ```

When VLM is disabled, the system falls back to Tesseract OCR automatically. When the reasoner or verifier is unavailable, the generator falls back to the configured provider or mock mode.

## LLM Provider Switching

The draft generator, reasoner, and verifier all use OpenAI-compatible endpoints. To switch backends without changing code, set one of these in `.env`:

```env
LLM_PROVIDER=lmstudio
# or
LLM_PROVIDER=ollama
# or
LLM_PROVIDER=openai
```

For local providers, keep the relevant `*_API_BASE` pointed at your OpenAI-compatible server. If you leave `LLM_PROVIDER` blank, the app keeps the previous auto-selection behavior for legacy calls.

---

## Docker (Full Stack)

```bash
# Copy and set your OpenAI key
copy .env.example .env
# Edit .env → set OPENAI_API_KEY

docker-compose up --build
# API: http://localhost:8000
# PostgreSQL: localhost:5432
```

## Full Local Integration Test

Use this when LM Studio, Docker, Tesseract, and Neo4j are already running locally.

```powershell
$env:Path = 'C:\Program Files\Tesseract-OCR;' + $env:Path
$env:VLM_ENABLED = 'true'
$env:VLM_API_BASE = 'http://localhost:1234/v1'
$env:VLM_MODEL = 'qwen/qwen3-vl-4b'
$env:REASONING_API_BASE = 'http://localhost:1234/v1'
$env:REASONING_MODEL = 'deepseek-r1-distill-llama-70b'
$env:VERIFIER_API_BASE = 'http://localhost:1234/v1'
$env:VERIFIER_MODEL = 'llama-4-8b'
$env:NEO4J_ENABLED = 'true'
$env:NEO4J_URI = 'bolt://localhost:7687'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = 'gldis_secret'

python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then, in a second terminal, upload a sample PDF and start processing:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/documents/upload -Form @{ file = Get-Item .\samples\sample_01_commercial_agreement.txt }
```

If you want a quick smoke test without uploading a new file, the unit suite that covers the current changes is:

```powershell
python -m pytest tests/test_ocr.py tests/test_retrieval.py tests/test_generation.py -q
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload a document (PDF/PNG/JPEG/TXT) |
| `POST` | `/api/documents/{id}/process/sync` | Process synchronously (blocks) |
| `POST` | `/api/documents/{id}/process` | Process in background |
| `GET`  | `/api/documents/{id}` | Get status, metadata, extracted fields |
| `GET`  | `/api/documents` | List all documents |
| `DELETE`| `/api/documents/{id}` | Delete document + indices |
| `POST` | `/api/drafts/generate` | Generate a grounded draft |
| `GET`  | `/api/drafts/{draft_id}` | Retrieve a draft |
| `GET`  | `/api/drafts/{draft_id}/evidence` | Get evidence used for a draft |
| `POST` | `/api/search` | Search evidence across documents |
| `POST` | `/api/feedback` | Submit an operator edit |
| `GET`  | `/api/feedback/history` | View recent feedback |
| `GET`  | `/api/feedback/improvements` | View learned rules + trend |
| `GET`  | `/api/feedback/examples` | View few-shot example pool |
| `GET`  | `/health` | Health check |
| `GET`  | `/api/status` | System status (VLM, vectors, model) |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **VLM** | InternVL2.5 / Qwen2.5-VL | Block-guided document understanding |
| **OCR Fallback** | PyMuPDF · Tesseract · PaddleOCR | Scanned/digital PDF extraction |
| **Image Preprocessing** | OpenCV · Pillow · pdf2image | Deskew, denoise, CLAHE, thresholding |
| **Chunking** | Custom semantic chunker | Section-aware, 400-token chunks |
| **NER** | spaCy `en_core_web_sm/lg` | Person, Org, Date, Money, Law |
| **Dense Retrieval** | FAISS `IndexFlatIP` + BGE-large | Cosine similarity search |
| **Sparse Retrieval** | BM25Okapi (rank-bm25) | Keyword + clause number search |
| **Graph Retrieval** | Neo4j GraphRAG | 1–2 hop neighborhood expansion |
| **Reasoner** | DeepSeek-R1-Distill-Llama-70B | Draft generation and repair |
| **Verifier** | Llama-4-8B | Correction list and grounding check |
| **Memory** | Mem0 | Feedback memory and preference capture |
| **Export** | JSONL DPO export | Offline preference tuning |
| **Backend** | FastAPI + SQLAlchemy | REST API + ORM |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Document, chunk, draft, feedback storage |
| **Frontend** | Vanilla JS + CSS | Zero-build UI with evidence highlighting |
| **Containerisation** | Docker + docker-compose | PostgreSQL + Redis + API |

---

## Running Tests

```bash
pytest tests/ -v
# Or by module:
pytest tests/test_extraction.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
pytest tests/test_feedback.py -v
```

For the current implementation slice, the most relevant local checks are:

```bash
pytest tests/test_ocr.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
```

---

## Evaluation

```bash
python evaluation/evaluate.py
```

Produces a report covering:
- **OCR quality** — CER / WER on sample documents
- **Retrieval quality** — Precision@K, Recall@K, F1@K, MRR
- **Grounding quality** — sentence-level grounding score, citation count
- **Human review framework** — usefulness / clarity / factual_consistency (0–5)
- **Improvement trend** — edit similarity delta over time

---

## Architecture Deep-Dive

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation of each stage.

---

## Design Decisions & Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **VLM-first, OCR-fallback** | VLM extracts text + structure + entities in one pass, eliminating a separate layout-parsing step for image documents |
| **Sentence-level grounding score** | Post-hoc Jaccard overlap is fast and requires no NLI model; acceptable tradeoff for a legal assessment pipeline |
| **SQLite default** | Zero infrastructure for local dev; swap `DATABASE_URL` to PostgreSQL for production |
| **Mock LLM fallback** | System always produces a structured output even with no API key, so the UI is always demonstrable |
| **RRF over learned fusion** | RRF requires no training data and consistently outperforms weighted fusion in zero-shot settings |
| **Vanilla JS UI** | Zero build step — the UI runs directly from FastAPI's static file server, no npm/node required |

---

## Project Structure

```
.
├── api/                # (reserved for future API versioning)
├── configs/            # Settings reference files
├── core/               # Shared config, schemas
├── db/                 # SQLAlchemy models + session
├── docker/             # (docker files at project root for simplicity)
├── evaluation/         # Metrics: CER/WER, Recall@K, grounding, human review
├── extraction/         # Rule-based regex + spaCy NER
├── feedback/           # Diff analyzer, feedback engine, improvement loop
├── generation/         # Prompts, grounding verifier, LLM generator
├── ingestion/          # Pipeline orchestrator
├── ocr/                # VLM extractor + Hybrid OCR dispatcher
├── preprocessing/      # Image pipeline + layout parser + semantic chunker
├── retrieval/          # FAISS store + BM25 store + Hybrid retriever
├── routes/             # FastAPI routers (documents, drafts, feedback, search)
├── samples/            # Sample legal documents for testing
├── static/             # Vanilla JS frontend (index.html, app.js, styles.css)
├── tests/              # Pytest test suite (unit + integration)
├── vlm/                # VLMParser class (wraps ocr/vlm_extractor)
├── .env.example        # Environment variable template
├── docker-compose.yml  # PostgreSQL + Redis + API
├── Dockerfile          # Production container image
├── main.py             # FastAPI application entrypoint
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
