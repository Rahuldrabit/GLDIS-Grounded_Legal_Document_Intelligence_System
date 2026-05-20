# GLDIS Architecture

This document describes the *implemented* architecture of **GLDIS (Grounded Legal Document Intelligence System)** as it exists in this repository. It focuses on runtime components, data flow, and where to find the code.

## Runtime entrypoint

- **Backend:** `main.py` (FastAPI)
  - Creates data directories and DB tables on startup (`core.config.Settings.ensure_dirs()`, `db.session.create_tables()`).
  - Mounts API routers from `routes/`.
  - Serves the built-in UI from `static/` when present.

> Note: There is also an `api/` package containing another FastAPI app (`api/main.py`). The actively used app (tests, README quickstart) is `main.py` + `routes/`. Treat `api/` as *legacy/prototype* unless you intentionally run it.

---

## 1. Full Pipeline Overview

End-to-end flow from upload through indexing, including all fallback paths and optional stages.

```mermaid
flowchart TD
  UPLOAD["POST /api/documents/upload\nroutes/documents.py"]
  PERSIST_FILE["Persist: data/uploads/uuid.ext\n+ DB row: documents table"]
  PROCESS["POST /api/documents/{id}/process\nasync background task\nor /process/sync blocking"]
  ORCH["run_pipeline()\ningestion/orchestrator.py"]

  UPLOAD --> PERSIST_FILE --> PROCESS --> ORCH

  ORCH --> PDF_CHECK{"is PDF?"}
  PDF_CHECK -->|Yes| PDF_IMGS["pdf_to_images() @300 DPI\npreprocessing/pipeline.py"]
  PDF_CHECK -->|Image or TXT| RAW["image_paths = [file_path]"]

  PDF_IMGS --> PDF_FB{"pdf2image\navailable?"}
  PDF_FB -->|Yes| P2I["convert_from_path()"]
  PDF_FB -->|"No + PyMuPDF\navailable"| FITZ["fitz pixel render"]
  PDF_FB -->|Neither available| NO_IMGS["image_paths = []\nstage continues"]

  P2I --> DOC_ROUTE
  FITZ --> DOC_ROUTE
  NO_IMGS --> DOC_ROUTE
  RAW --> DOC_ROUTE

  DOC_ROUTE{"Document Understanding\nRouting"}
  DOC_ROUTE -->|"is_vlm_blocks_available()\n+ image_paths present"| BLOCK_VLM["extract_vlm_with_blocks()\nocr/block_extraction_adapter.py"]
  DOC_ROUTE -->|Otherwise| HYBRID_OCR["HybridOCR.extract()\nocr/hybrid_ocr.py\n(see OCR diagram)"]

  BLOCK_VLM --> PERSIST_OCR
  HYBRID_OCR --> PERSIST_OCR

  PERSIST_OCR["Persist OCR artifacts\nartifacts/doc_id/ocr.json\nDB: vlm_query_logs + ocr_blocks\ntry/except - non-fatal"]

  PERSIST_OCR --> TEXT_CHECK{"Any text\nextracted?"}
  TEXT_CHECK -->|No| FAILED["Mark FAILED\nDocumentStatus.failed\n+ error_message"]
  TEXT_CHECK -->|Yes| LAYOUT["Layout parsing\npreprocessing/layout_parser.py\ntry/except - non-fatal, layout=None"]

  LAYOUT --> CHUNK_ROUTE{"layout AND\nlayout.blocks?"}
  CHUNK_ROUTE -->|Yes| CHUNK_LAYOUT["chunk_layout()\nsection-aware packing\npreprocessing/chunker.py"]
  CHUNK_ROUTE -->|No| CHUNK_TEXT["chunk_text()\nflat sentence splitting\npreprocessing/chunker.py"]

  CHUNK_LAYOUT --> PERSIST_CHUNKS
  CHUNK_TEXT --> PERSIST_CHUNKS

  PERSIST_CHUNKS["Persist chunks\nDB: chunks\nartifacts/doc_id/chunks.json\ntry/except - non-fatal"]

  PERSIST_CHUNKS --> EXTRACT["Rule-based extraction\nextraction/rule_based.py\nregex: dates, case nos, money"]
  EXTRACT --> NER_CHECK{"spaCy\navailable?"}
  NER_CHECK -->|"en_core_web_lg\nloaded OK"| NER_LG["NER via en_core_web_lg"]
  NER_CHECK -->|"lg OSError\nen_core_web_sm"| NER_SM["NER via en_core_web_sm"]
  NER_CHECK -->|"both fail\nor ImportError"| NER_OFF["NER disabled\ngracefully"]

  NER_LG --> PERSIST_FIELDS
  NER_SM --> PERSIST_FIELDS
  NER_OFF --> PERSIST_FIELDS

  PERSIST_FIELDS["Persist structured_fields\nDB: structured_fields"]

  PERSIST_FIELDS --> GRAPH_IDX_CHECK{"graphrag_enabled\n+ neo4j_enabled?"}
  GRAPH_IDX_CHECK -->|"Yes + driver\nconnected"| NEO4J_IDX["index_blocks_in_graph()\nNeo4j Cypher\nretrieval/graph_store.py\ntry/except - non-fatal"]
  GRAPH_IDX_CHECK -->|"Disabled or\nconnection fail"| GRAPH_SKIP["skip - non-fatal\nlog warning"]

  NEO4J_IDX --> INDEX
  GRAPH_SKIP --> INDEX

  INDEX["add_chunks()\nretrieval/hybrid_retriever.py\ntry/except - non-fatal"]
  INDEX --> FAISS_IDX["FAISS dense index\nretrieval/vector_store.py\ndata/faiss_index/"]
  INDEX --> BM25_IDX["BM25 sparse index\nretrieval/bm25_retriever.py\ndata/bm25_index/"]

  FAISS_IDX --> READY["Mark READY\nDocumentStatus.ready"]
  BM25_IDX --> READY
```

---

## 2. OCR Routing Detail

All engine availability is detected at runtime via try-import checks. Engine selection cascades through fallbacks automatically.

```mermaid
flowchart TD
  OCR_IN["HybridOCR.extract()\nocr/hybrid_ocr.py"]

  OCR_IN --> FORCE_CHECK{"force_engine\nparam set?"}
  FORCE_CHECK -->|"PYMUPDF"| PYMUPDF_FORCED["_run_pymupdf()"]
  FORCE_CHECK -->|"TESSERACT"| TESS_FORCED["_run_tesseract()"]
  FORCE_CHECK -->|"PADDLEOCR"| PADDLE_FORCED["_run_paddleocr()"]
  FORCE_CHECK -->|"VLM"| VLM_FORCED["_run_vlm()"]
  FORCE_CHECK -->|None - auto-route| AUTO

  AUTO{"Auto routing"}
  AUTO -->|"is_pdf\n+ has_pymupdf\n+ _is_digital_pdf()"| PYMUPDF["_run_pymupdf()\nconf = 1.0"]
  AUTO -->|"Scanned PDF\nor image input"| VLM_ROUTE{"vlm_enabled=True\n+ image_paths present?"}

  VLM_ROUTE -->|Yes| MINERU_CHECK{"mineru_enabled\n=True?"}
  VLM_ROUTE -->|No| OCR_FB

  MINERU_CHECK -->|Yes| MINERU["extract_layout_blocks_mineru()\nMinerU CLI subprocess\npreprocessing/pipeline.py"]
  MINERU_CHECK -->|No| STD_VLM["_run_vlm()\nfull-page VLM"]

  MINERU -->|"Blocks returned"| BLOCK_VLM["extract_blocks_with_vlm()\nblock-guided VLM per region"]
  MINERU -->|"Exception or returncode != 0\nor JSON output missing"| DUMMY["Create dummy full-page blocks\nfrom image PIL dimensions\ntry/except per image"]

  DUMMY --> BLOCK_VLM

  BLOCK_VLM --> BCONF{"avg_conf >=\nvlm_confidence_threshold?"}
  BCONF -->|Yes| OCR_DONE["PageOCRResult[]\nwith block grounding"]
  BCONF -->|"No - low confidence"| STD_VLM

  STD_VLM --> VCONF{"avg_conf >=\nvlm_confidence_threshold?"}
  VCONF -->|Yes| OCR_DONE
  VCONF -->|"No - low confidence"| OCR_FB

  OCR_FB["_ocr_fallback()"]
  OCR_FB --> TESS_AVAIL{"has_tesseract?"}
  TESS_AVAIL -->|Yes| TESS["Tesseract OCR\npytesseract\nper-page try/except"]
  TESS_AVAIL -->|"No + has_paddleocr"| PADDLE["PaddleOCR\nper-page try/except"]
  TESS_AVAIL -->|"Neither available"| EMPTY["Return []\nlog error: no engine"]

  TESS --> TCONF{"avg_conf >= 0.6\n+ has_paddleocr?"}
  TCONF -->|"conf OK"| OCR_DONE
  TCONF -->|"Low conf"| PADDLE

  PADDLE --> OCR_DONE

  PYMUPDF --> OCR_DONE
  PYMUPDF_FORCED --> OCR_DONE
  TESS_FORCED --> OCR_DONE
  PADDLE_FORCED --> OCR_DONE
  VLM_FORCED --> OCR_DONE
```

---

## 3. LLM Provider Resolution and Draft Generation

Provider selection, verifier loop, and grounding pipeline.

```mermaid
flowchart TD
  GEN_IN["POST /api/drafts/generate\nroutes/drafts.py"]

  GEN_IN --> RETRIEVE_EV["Retrieve evidence\nHybridRetriever.search()\n(see Retrieval diagram)"]
  RETRIEVE_EV --> TRIM_EV["Trim to max_evidence_tokens budget\ngeneration/generator.py"]
  TRIM_EV --> FB_CTX["Get improvement context\nfeedback/improvement_loop.py\nfew-shot examples + style rules"]

  FB_CTX --> REASONER_CHECK{"reasoning_model\n+ reasoning_api_base\nboth non-empty?"}

  REASONER_CHECK -->|Yes| REASONER["call_reasoner()\nDeepSeek-R1\nDEDICATED HTTP endpoint"]
  REASONER_CHECK -->|No| LLM_RESOLVE

  REASONER -->|"HTTP success"| JSON_TRY{"Parse JSON\nchoices[0].message.content"}
  REASONER -->|"Exception / HTTP fail"| LLM_RESOLVE

  JSON_TRY -->|"Parse OK"| LLM_OUT
  JSON_TRY -->|"JSON parse fail"| RAW_RESP["Return raw response string"]
  RAW_RESP --> LLM_OUT

  LLM_RESOLVE["resolve_llm_config(mode)\nllm/client.py"]

  LLM_RESOLVE --> EXPLICIT_CHECK{"llm_provider\nexplicit in config?"}
  EXPLICIT_CHECK -->|Yes| NORMALIZE["Normalize:\nlmstudio/lm-studio/local -> lmstudio\nollama -> ollama\nopenai -> openai\nunknown -> lmstudio"]
  EXPLICIT_CHECK -->|No| AUTODETECT{"Auto-detect"}

  AUTODETECT -->|"mode=vision\n+ vlm_enabled"| USE_LMSTUDIO["lmstudio\n(local endpoint)"]
  AUTODETECT -->|"openai_api_key\nnon-empty + not placeholder"| USE_OPENAI["openai\napi.openai.com/v1"]
  AUTODETECT -->|default| USE_LMSTUDIO

  NORMALIZE --> CHAT_CALL
  USE_LMSTUDIO --> CHAT_CALL
  USE_OPENAI --> CHAT_CALL

  CHAT_CALL["chat_completion()\nllm/client.py"]
  CHAT_CALL -->|"HTTP 200 OK"| LLM_OUT["LLM response text"]
  CHAT_CALL -->|"All providers raise"| MOCK["_call_mock()\ntemplate generator\nno LLM required"]
  MOCK --> LLM_OUT

  LLM_OUT --> VERIFIER_LOOP{"verifier_enabled=True\n+ iter < max_correction_iterations?"}
  VERIFIER_LOOP -->|No| GROUNDING_STEP
  VERIFIER_LOOP -->|Yes| VERIFY["verify_draft()\nverifier endpoint\nverifier_model + verifier_api_base"]

  VERIFY -->|"Empty corrections list"| GROUNDING_STEP
  VERIFY -->|"Corrections returned"| RECORRECT["Re-call _call_llm()\nwith correction message appended\niter_count++"]
  RECORRECT --> VERIFIER_LOOP

  GROUNDING_STEP["compute_grounding_score()\ngeneration/grounding.py\nextract_citations()"]
  GROUNDING_STEP --> UNGRD_CHECK{"Ungrounded\nsentences found?"}
  UNGRD_CHECK -->|Yes| ANNOTATE["_annotate_ungrounded()\nappend warning section\nup to 5 sentences listed"]
  UNGRD_CHECK -->|No| DRAFT_PERSIST
  ANNOTATE --> DRAFT_PERSIST

  DRAFT_PERSIST["Persist draft\nDB: drafts\nartifacts/doc_id/draft.json\ntry/except - non-fatal"]
```

---

## 4. Retrieval Pipeline

Hybrid dense+sparse search with optional reranking and GraphRAG expansion.

```mermaid
flowchart TD
  QINPUT["HybridRetriever.search(query, top_k, doc_filter)\nretrieval/hybrid_retriever.py"]

  QINPUT --> EMBED["Embed query\nembedding_model"]
  QINPUT --> BM25Q["BM25 tokenize\nretrieval/bm25_retriever.py"]

  EMBED --> FAISS_SEARCH["FAISS dense search\nretrieval/vector_store.py\ntop_k candidates"]
  BM25Q --> BM25_SEARCH["BM25 sparse search\ntop_k candidates"]

  FAISS_SEARCH --> DOC_FILT1{"document_id\nfilter set?"}
  BM25_SEARCH --> DOC_FILT2{"document_id\nfilter set?"}

  DOC_FILT1 -->|Yes| FILT_DENSE["filter results by doc_id"]
  DOC_FILT1 -->|No| ALL_DENSE["all results"]
  DOC_FILT2 -->|Yes| FILT_SPARSE["filter results by doc_id"]
  DOC_FILT2 -->|No| ALL_SPARSE["all results"]

  FILT_DENSE --> RRF
  ALL_DENSE --> RRF
  FILT_SPARSE --> RRF
  ALL_SPARSE --> RRF

  RRF["Reciprocal Rank Fusion\ndense_weight=0.6  sparse_weight=0.4"]

  RRF --> RERANK_CHECK{"use_reranker=True\n+ candidates > rerank_top_k?"}
  RERANK_CHECK -->|No| TOPK_SLICE["top-k slice"]
  RERANK_CHECK -->|Yes| LOAD_CE{"CrossEncoder\nloaded OK?"}

  LOAD_CE -->|"sentence_transformers\navailable"| CE_RERANK["CrossEncoder.predict()\ncross-encoder/ms-marco-MiniLM-L-6-v2\nlazy-loaded singleton"]
  LOAD_CE -->|"ImportError or init fail"| RERANK_SKIP["return candidates[:top_k]\nlog warning"]

  CE_RERANK -->|"predict() success"| RERANKED["Reranked results"]
  CE_RERANK -->|"Exception"| RERANK_SKIP

  RERANKED --> GRAPH_EXPAND_CHECK
  TOPK_SLICE --> GRAPH_EXPAND_CHECK
  RERANK_SKIP --> GRAPH_EXPAND_CHECK

  GRAPH_EXPAND_CHECK{"graphrag_enabled=True\n+ graph_store.enabled?"}
  GRAPH_EXPAND_CHECK -->|"Yes + Neo4j\nconnected"| NEO4J_EXP["expand_chunk_neighborhood()\ntop_chunk_ids\nhops=graph_expand_hops\nNeo4j Cypher"]
  GRAPH_EXPAND_CHECK -->|"Disabled or\nnot connected"| RET_OUT

  NEO4J_EXP -->|"Cypher success"| MERGE_DEDUP["Merge + deduplicate\nexpanded chunks\ncheck _meta_cache + faiss_store._meta"]
  NEO4J_EXP -->|"Exception"| GRAPH_WARN["log warning\nskip - non-fatal"]

  MERGE_DEDUP --> RET_OUT
  GRAPH_WARN --> RET_OUT

  RET_OUT["Evidence chunks\nList[ChunkResult] with scores"]
```

---

## 5. Feedback and Memory Storage

```mermaid
flowchart TD
  FB_IN["POST /api/feedback\nroutes/feedback.py"]

  FB_IN --> PERSIST_EDIT["Persist edit\nDB: edits table"]
  PERSIST_EDIT --> PROMOTE_CHECK{"High-quality edit?\nquality threshold"}
  PROMOTE_CHECK -->|Yes| FEW_SHOT["Promote to\nfew_shot_examples table\nused as future generation context"]
  PROMOTE_CHECK -->|No| MEM_WRITE
  FEW_SHOT --> MEM_WRITE

  MEM_WRITE{"Mem0 client\ninitialized?"}
  MEM_WRITE -->|"mem0 installed\n+ Client() OK"| MEM0_SAVE["mem0.Client.save(record)\nfeedback/mem0_store.py"]
  MEM_WRITE -->|"ImportError\nor init fail"| JSONL_SAVE["Append to JSONL file\nfeedback_store_path/mem0.jsonl"]

  MEM0_SAVE -->|"save() OK"| FB_DONE
  MEM0_SAVE -->|"Exception"| JSONL_SAVE
  JSONL_SAVE --> FB_DONE

  FB_DONE["Feedback stored"]

  READ_CTX["get_generation_context()\nfeedback/improvement_loop.py"]
  READ_CTX --> GET_EXAMPLES["_get_best_examples()\nquery DB few_shot_examples\noptional feedback_type_filter"]
  GET_EXAMPLES --> SELECT_DIVERSE["_select_diverse()\nif candidates <= n: return all\nelse: diversity selection"]
  SELECT_DIVERSE --> AGG_STYLE["_aggregate_style_rules()\ncount >= 1 threshold to include rule"]
  AGG_STYLE --> CTX_OUT["Generation context:\nfew-shot examples + style rules"]
```

---

## Pipeline stages

### 1) Upload (API)
- **Route:** `POST /api/documents/upload` (`routes/documents.py`)
- Persists:
  - File → `./data/uploads/<uuid>.<ext>`
  - DB row → `documents` table

### 2) Process / ingest (pipeline)
- **Routes:**
  - `POST /api/documents/{id}/process` (background task)
  - `POST /api/documents/{id}/process/sync` (blocking)
- **Implementation:** `ingestion/orchestrator.run_pipeline()`

What happens:
- **Preprocessing:** PDF → page images at 300 DPI (best-effort) (`preprocessing/pipeline.py`).
- **HybridOCR routing:** (`ocr/hybrid_ocr.py`)
  - If PDF has a text layer → **PyMuPDF** extraction.
  - Otherwise, if `vlm_enabled=true` and images are available → **VLM-first**, then OCR fallback if confidence is low.
  - OCR fallback chain: **Tesseract → PaddleOCR**.
- **Persistence / artifacts:**
  - Writes `ocr.json`, `layout.json`, `chunks.json` under `./data/artifacts/<document_id>/` (best-effort).
  - Stores VLM audit logs in `vlm_query_logs`.
  - Stores block-level results in `ocr_blocks` and links chunk→block grounding when possible.

### 3) Layout parsing + chunking
- **Layout parsing:** `preprocessing/layout_parser.py` (rule-based heading/clause detection; page markers `--- PAGE N ---` are used when present).
- **Chunking:** `preprocessing/chunker.py`
  - Section-aware packing to a target token budget (approx token counting) with overlap.

### 4) Extraction (structured fields)
- **Rule-based:** `extraction/rule_based.py` (regex patterns for dates, case numbers, money, etc.)
- **NER:** `extraction/ner_extractor.py` (spaCy; disabled gracefully if model not installed)
- Persisted to `structured_fields`.

### 5) Indexing + retrieval
- **Indexing:** `retrieval/hybrid_retriever.HybridRetriever.add_chunks()`
  - **FAISS** (dense) persists under `./data/faiss_index/`.
  - **BM25** (sparse) persists under `./data/bm25_index/`.
- **Query-time retrieval:** `HybridRetriever.search()`
  - Pulls candidates from FAISS + BM25
  - Fuses rankings with **Reciprocal Rank Fusion (RRF)**
  - Optional cross-encoder reranking (lazy-loaded)
  - Optional **GraphRAG expansion** if enabled (see below)

### 6) Draft generation + grounding
- **Route:** `POST /api/drafts/generate` (`routes/drafts.py`)
- Steps:
  1. Retrieve evidence via `HybridRetriever.search()`.
  2. Fetch improvement context (few-shot + style rules) via `feedback/improvement_loop.py`.
  3. Generate using `generation/generator.py`:
     - Prefer a dedicated **reasoner endpoint** if configured.
     - Otherwise use the resolved provider in `llm/client.py`.
     - If all providers fail → mock structured template output (keeps system demoable).
  4. Optional verifier loop (bounded by `max_correction_iterations`).
  5. Compute grounding score + extract inline citations (`generation/grounding.py`).

### 7) Feedback → improvements
- **Route:** `POST /api/feedback` (`routes/feedback.py`)
- Persists edits in `edits` and may promote high-quality edits into `few_shot_examples`.
- Optional Mem0 integration exists (`feedback/mem0_store.py`) with JSONL fallback.

---

## Data storage (tables you should expect)
Defined in `db/models.py`:
- `documents`, `ocr_blocks`, `chunks`, `structured_fields`
- `drafts`, `edits`, `few_shot_examples`, `vlm_query_logs`

---

## Feature flags / configuration
Defined in `core/config.py` (loaded from `.env`):

| Flag | Type | Default | Controls |
|---|---|---|---|
| `vlm_enabled` | bool | `False` | VLM-first routing; `is_vlm_blocks_available()`; vision mode in provider resolution |
| `vlm_confidence_threshold` | float | `0.6` | Threshold to fall back from VLM to OCR |
| `mineru_enabled` | bool | `False` | MinerU CLI subprocess for block layout extraction |
| `graphrag_enabled` | bool | `False` | GraphRAG neighborhood expansion in HybridRetriever |
| `neo4j_enabled` | bool | `False` | Neo4j driver initialization in GraphStore |
| `verifier_enabled` | bool | `True` | Auto-correction loop in DraftGenerator |
| `max_correction_iterations` | int | `2` | Max verifier→correction cycles |
| `reasoning_model` + `reasoning_api_base` | str | `""` | Enables DeepSeek-R1 reasoner path (both must be non-empty) |
| `verifier_model` + `verifier_api_base` | str | `""` | Enables dedicated verifier API endpoint |
| `llm_provider` | str | `""` | Explicit provider override (empty = auto-detect) |
| `openai_api_key` | str | `""` | Triggers OpenAI provider if non-empty and not placeholder |
| `graph_expand_hops` | int | `1` | Neo4j traversal depth for neighborhood expansion |
| `retrieval_top_k` | int | `10` | Default candidates from each index |
| `rerank_top_k` | int | `5` | Final results after fusion/reranking |
| `chunk_target_tokens` | int | `400` | Chunking size threshold |
| `chunk_overlap_tokens` | int | `75` | Chunk overlap for context continuity |
| `max_evidence_tokens` | int | `3000` | Evidence budget trim before generation |
| `database_url` | str | `sqlite:///./data/gldis.db` | Database connection |
| `embedding_model` | str | set | Sentence transformer for FAISS embeddings |
| `faiss_index_path` | str | `./data/faiss_index` | FAISS persistence path |
| `bm25_index_path` | str | `./data/bm25_index` | BM25 persistence path |

---

## Repo map (high-signal folders)
- `routes/` — public API used by `main.py`
- `ingestion/`, `ocr/`, `preprocessing/`, `extraction/`, `retrieval/`, `generation/`, `feedback/` — pipeline modules
- `static/` — built-in UI served by backend
- `ui/` — optional React/Vite dev UI

---
