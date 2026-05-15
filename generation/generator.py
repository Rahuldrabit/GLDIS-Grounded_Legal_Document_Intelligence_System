"""
Stage 10 — Grounded Draft Generator
Uses RAG to generate Case Fact Summaries and Internal Memos
anchored to retrieved evidence. Uses the shared LLM provider helper.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from core.config import get_settings
from llm.client import chat_completion, resolve_llm_config
from generation.reasoner import call_reasoner
from core.schemas import (
    Citation, DraftRequest, DraftResponse, EvidenceChunk, StructuredExtraction
)
from generation.grounding import compute_grounding_score, extract_citations
from generation.prompts import build_generation_prompt
from generation.verifier import verify_draft

import logging

logger = logging.getLogger(__name__)


def _call_mock(messages: list, evidence_chunks: List[EvidenceChunk]) -> str:
    """
    Fallback mock generator — produces a structured template using
    only the evidence text. Used when no LLM is available.
    """
    logger.warning("Using mock generator — no LLM configured.")
    evidence_text = "\n".join(
        f"- {c.text[:200]}... [Source: {c.chunk_id}, p.{c.page or '?'}]"
        for c in evidence_chunks[:5]
    )
    return f"""## CASE FACT SUMMARY

### Parties Involved
- Insufficient evidence to determine parties without LLM processing.

### Key Facts
{evidence_text}

### Key Dates & Deadlines
- Insufficient evidence to determine without LLM processing.

---

## INTERNAL MEMO

**RE:** Document Review — AI-Assisted Summary
**DATE:** {datetime.utcnow().strftime('%Y-%m-%d')}
**FROM:** AI Legal Assistant (Mock Mode)
**TO:** Reviewing Attorney

### Summary
This draft was generated in mock mode (no LLM configured). The {len(evidence_chunks)} evidence passages above were retrieved from the document. Please configure OPENAI_API_KEY or start Ollama to enable full generation.

### Evidence Gaps
- Full generation unavailable: configure an LLM provider.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main Generator
# ──────────────────────────────────────────────────────────────────────────────

class DraftGenerator:
    """
    Generates grounded Case Fact Summary + Internal Memo drafts.
    Routing order: configured provider → OpenAI → Mock
    """

    def __init__(self):
        self.settings = get_settings()

    def generate(
        self,
        request: DraftRequest,
        evidence_chunks: List[EvidenceChunk],
        structured_data: Optional[StructuredExtraction] = None,
        few_shot_examples: Optional[list] = None,
        style_rules: Optional[List[str]] = None,
    ) -> DraftResponse:
        """
        Generate a grounded draft from evidence chunks.
        
        Args:
            request: DraftRequest with document_id and query.
            evidence_chunks: Retrieved evidence passages.
            structured_data: Optional pre-extracted entities.
            few_shot_examples: High-quality examples from feedback loop.
            style_rules: Aggregated style preferences from operator edits.
        """
        # Truncate evidence to stay within token budget
        evidence_chunks = self._trim_evidence(evidence_chunks)

        # Build structured fields dict for prompt injection
        sf_dict = None
        if structured_data:
            sf_dict = {
                "Case Number": structured_data.case_number,
                "Plaintiff": structured_data.plaintiff,
                "Defendant": structured_data.defendant,
                "Dates": structured_data.dates[:5],
                "Monetary Values": structured_data.monetary_values[:5],
                "Parties": structured_data.parties[:5],
                "Deadlines": structured_data.deadlines[:3],
            }

        # Build messages
        messages = build_generation_prompt(
            query=request.query or "Generate a case fact summary and internal memo.",
            evidence_chunks=evidence_chunks,
            few_shot_examples=few_shot_examples,
            style_rules=style_rules,
            structured_fields=sf_dict,
        )

        # Call LLM (initial pass)
        generated_text = self._call_llm(messages, evidence_chunks)

        # Optional verifier + bounded auto-correction loop
        corrections = []
        max_iters = self.settings.max_correction_iterations or 0
        iter_count = 0
        while (
            self.settings.verifier_enabled
            and iter_count < max_iters
        ):
            iter_count += 1
            corrections = verify_draft(generated_text, evidence_chunks)
            if not corrections:
                break

            # Instruct the reasoner to apply corrections and regenerate
            correction_msg = (
                "The verifier found the following issues. Apply the suggested fixes "
                "to the draft while still obeying the grounding rules. Respond with the full revised draft.\n\n"
                + "Corrections JSON:\n"
                + str(corrections)
            )
            messages_with_corrections = list(messages)
            messages_with_corrections.append({"role": "user", "content": correction_msg})
            generated_text = self._call_llm(messages_with_corrections, evidence_chunks)

        # Post-process: grounding score + citation extraction
        grounding_score, ungrounded = compute_grounding_score(
            generated_text, evidence_chunks
        )
        citations = extract_citations(generated_text, evidence_chunks)

        # Annotate ungrounded sentences
        if ungrounded:
            generated_text = self._annotate_ungrounded(generated_text, ungrounded)

        draft_id = str(uuid.uuid4())
        logger.info(
            f"Draft {draft_id} generated | "
            f"grounding={grounding_score:.2f} | "
            f"citations={len(citations)} | "
            f"ungrounded={len(ungrounded)}"
        )

        draft_response = DraftResponse(
            draft_id=draft_id,
            document_id=request.document_id,
            generated_text=generated_text,
            citations=citations,
            evidence_chunks=evidence_chunks,
            grounding_score=grounding_score,
        )

        try:
            import json
            from pathlib import Path
            artifact_dir = Path(self.settings.upload_dir).parent / "artifacts" / request.document_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with open(artifact_dir / "draft.json", "w", encoding="utf-8") as f:
                f.write(draft_response.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to save draft.json: {e}")

        return draft_response

    def _call_llm(self, messages: list, evidence_chunks: List[EvidenceChunk]) -> str:
        """Try the configured provider → OpenAI → Mock in order."""
        # Prefer a dedicated reasoner endpoint if configured (stage 3)
        settings = self.settings
        try:
            if getattr(settings, "reasoning_model", None) and getattr(settings, "reasoning_api_base", None):
                logger.info(
                    f"Calling reasoner model={settings.reasoning_model} base={settings.reasoning_api_base}"
                )
                raw = call_reasoner(messages, temperature=settings.generation_temperature, max_tokens=settings.llm_max_tokens)
                # When using the reasoner endpoint, the API response may be raw JSON/text.
                # Try to extract content field if present (OpenAI-like wrapper)
                # Fallback: return full raw response
                try:
                    import json

                    parsed = json.loads(raw)
                    # openai-like: choices[0].message.content
                    content = parsed.get("choices", [])[0].get("message", {}).get("content")
                    if content:
                        return content
                except Exception:
                    return raw

            # Otherwise fall back to configured LLM provider
            config = resolve_llm_config(mode="text")
            logger.info(
                f"Calling LLM provider={config.provider} model={config.model} base_url={config.base_url}"
            )
            response = chat_completion(
                messages,
                mode="text",
                model=config.model,
                temperature=self.settings.generation_temperature,
                max_tokens=config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"Configured LLM call failed: {exc}. Using mock generator.")

        # Fallback to mock
        return _call_mock(messages, evidence_chunks)

    def _trim_evidence(self, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        """Trim evidence to stay within max_evidence_tokens budget."""
        settings = self.settings
        budget = settings.max_evidence_tokens * 4  # ~4 chars/token
        total = 0
        kept = []
        for chunk in chunks:
            total += len(chunk.text)
            if total > budget:
                break
            kept.append(chunk)
        return kept

    def _annotate_ungrounded(self, text: str, ungrounded: List[str]) -> str:
        """Append an evidence gap warning section to the draft."""
        if not ungrounded:
            return text
        warning = "\n\n---\n\n### ⚠️ GROUNDING WARNINGS\nThe following sentences could not be directly verified against the provided evidence:\n"
        for sent in ungrounded[:5]:  # Cap at 5 warnings
            warning += f"\n- {sent[:150]}..."
        return text + warning
