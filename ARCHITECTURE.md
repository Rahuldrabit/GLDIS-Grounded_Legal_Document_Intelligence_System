# GLDIS Architecture — Technical Deep-Dive

## Overview

The Grounded Legal Document Intelligence System (GLDIS) is a 4-stage, modular pipeline. Each stage is independently testable, has clear input/output contracts defined by Pydantic schemas, and degrades gracefully when optional dependencies (MinerU, Neo4j, Mem0, model servers) are unavailable.

---

## Stage Map

```
┌─────────────────────────────────────────────────────────────────┐
│  [1] STAGE 1: INGEST + BLOCK EXTRACTION                         │
│                                                                 │
│  Upload API → preprocessing → OCR/VLM routing → MinerU optional │
│  InternVL2.5 block extraction → real pixel bboxes @ 300 DPI    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  [2] STAGE 2: RETRIEVAL                                         │
│                                                                 │
│  Semantic chunking → FAISS dense + BM25 sparse + GraphRAG       │
│  Neo4j 1–2 hop expansion → fused evidence chunks                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  [3] STAGE 3: REASON + VERIFY                                   │
│                                                                 │
│  DeepSeek-R1 reasoner → structured claim map → Llama-4 verifier│
│  bounded correction loop → grounded draft + citations          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  [4] STAGE 4: MEMORY + LEARNING                                 │
│                                                                 │
│  Mem0 feedback memory → operator edit capture → DPO JSONL export│
│  offline preference tuning dataset                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1 — Upload + block extraction

- Accepts PDF, PNG, JPEG, TIFF, BMP, TXT
- Assigns UUID, stores file, creates `Document` DB record
- Returns `document_id` for all subsequent calls
- Processing triggered via `POST /api/documents/{id}/process/sync`
- PDF preprocessing renders at 300 DPI; optional MinerU provides layout blocks
- VLM-guided extraction crops each block and returns real `bbox=[x1,y1,x2,y2]` values

### Stage 2 — Retrieval (`retrieval/`)

| Component | Technique | Notes |
|-----------|-----------|-------|
| Dense retrieval | FAISS `IndexFlatIP` | existing embeddings remain unchanged |
| Sparse retrieval | BM25 / token-overlap fallback | works even when `rank_bm25` is missing |
| Graph expansion | Neo4j + APOC | chunk neighborhood traversal for 1–2 hops |
| Fusion | Reciprocal Rank Fusion | combines dense, sparse, and graph-derived evidence |

### Stage 3 — Reasoning + verification (`generation/`)

- Reasoner endpoint: `DeepSeek-R1-Distill-Llama-70B` via an OpenAI-compatible server
- Verifier endpoint: `Llama-4-8B` via an OpenAI-compatible server
- Output contract: `draft_markdown` plus structured claim map for audit and repair
- Verification loop is bounded to 1–2 iterations and only rewrites the draft when the verifier finds unsupported claims

### Stage 4 — Memory + learning (`feedback/`)

- `Mem0Store` captures feedback memory when the optional package is present, otherwise falls back to JSONL persistence
- `dpo_export.py` emits edit pairs as JSONL for offline preference tuning
- Existing edit capture and diff analysis remain the source of training examples

**Category heuristics:**
| Edit type | Signal |
|-----------|--------|
| `hallucination` | Original had hedges (`likely`, `probably`), edited has specifics |
| `missing_fact` | Large insertion with dates/amounts/names |
| `incorrect_fact` | Replacement of specific values |
| `poor_structure` | Many insertions/deletions uniformly distributed |

---

## Data Models (`db/models.py`)

| Table | Key fields |
|-------|-----------|
| `documents` | id, filename, status, page_count |
| `chunks` | chunk_id, document_id, text, page, section, token_count |
| `ocr_blocks` | block_id, document_id, text, bbox, block_type |
| `structured_fields` | document_id, field, value, confidence |
| `drafts` | draft_id, document_id, generated_text, evidence_used, citations, grounding_score |
| `edits` | edit_id, draft_id, original_text, edited_text, edit_diff, feedback_type |
| `vlm_query_logs` | id, document_id, image_path, prompt, response, model_used |

---

## Schemas (`core/schemas.py`)

Key Pydantic models shared across all stages:


---

## Grounding Strategy (3 Layers)

| Layer | Where | Mechanism |
|-------|-------|-----------|
| **Generation-time** | Prompt | system rules + evidence-only instruction + structured claim map |
| **Evidence injection** | Prompt | chunked evidence with source and page headers |
| **Post-hoc verification** | `grounding.py` + verifier loop | Jaccard overlap score + verifier corrections + bounded rewrite |

---

## Scaling Considerations

| Concern | Current | Production upgrade |
|---------|---------|-------------------|
| Vector index | FAISS Flat (exact) | FAISS IVF or HNSW for >1M vectors |
| Database | SQLite | PostgreSQL (set `DATABASE_URL`) |
| Processing | Synchronous | Celery + Redis task queue |
| Embedding | In-process | Dedicated embedding microservice |
| LLM | Reasoner + verifier HTTP calls | batched async calls with semaphore |
