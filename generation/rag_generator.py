"""
Stage 10 — Grounded Draft Generation (RAG)
Generates case summaries and legal memos using ONLY retrieved evidence,
with explicit citation and hallucination controls.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from core.config import get_settings
from core.schemas import Citation, DraftResponse, EvidenceChunk
from llm.client import chat_completion, resolve_llm_config

logger = logging.getLogger(__name__)

# Prompt templates
SYSTEM_PROMPT = """You are an expert legal assistant drafting an internal case memo.
Your primary objective is GROUNDED GENERATION.
You must adhere STRICTLY to the following rules:

1. ONLY use facts from the provided Evidence sections.
2. If the required information is missing from the Evidence, you MUST explicitly state: "Information not found in the provided documents."
3. Do NOT hallucinate names, dates, amounts, or clauses.
4. For every factual claim you make, you MUST cite the source document inline using the format: [Source: chunk_id, page N].
5. Format the output professionally using Markdown headings and bullet points where appropriate.

Evidence Format provided to you:
[CHUNK_ID: <id>] (Page: <page>, Section: <section>)
<text>
"""


class RAGGenerator:
    """Retrieval-Augmented Generation using OpenAI (or compatible local LLM)."""

    def __init__(self):
        settings = get_settings()
        self.config = resolve_llm_config(mode="text")
        self.temperature = settings.generation_temperature

    def generate_draft(
        self,
        document_id: str,
        query: str,
        evidence: List[EvidenceChunk],
        few_shot_examples: Optional[List[dict]] = None
    ) -> DraftResponse:
        """
        Generate a drafted response grounded in the provided evidence.
        """
        # 1. Format Evidence
        evidence_text = self._format_evidence(evidence)
        
        # 2. Construct Prompt Messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add few-shot examples if provided (from feedback loop)
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": f"Evidence:\n{example['evidence']}\n\nTask:\n{example['query']}"})
                messages.append({"role": "assistant", "content": example['corrected_draft']})
                
        user_prompt = f"Evidence:\n{evidence_text}\n\nTask:\n{query}"
        messages.append({"role": "user", "content": user_prompt})

        logger.info(f"Generating draft for doc {document_id} with {len(evidence)} evidence chunks")

        # 3. Call LLM
        try:
            response = chat_completion(
                messages,
                mode="text",
                model=self.config.model,
                temperature=self.temperature,
                max_tokens=self.config.max_tokens,
            )
            generated_text = response.choices[0].message.content or ""
            
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            raise

        # 4. Extract Citations and calculate basic grounding score
        citations = self._extract_citations(generated_text, evidence)
        grounding_score = self._calculate_grounding_score(generated_text, citations, evidence)

        import uuid
        draft_id = str(uuid.uuid4())

        return DraftResponse(
            draft_id=draft_id,
            document_id=document_id,
            generated_text=generated_text,
            citations=citations,
            evidence_chunks=evidence,
            grounding_score=grounding_score,
        )

    def _format_evidence(self, evidence: List[EvidenceChunk]) -> str:
        """Format evidence chunks into a structured string for the LLM context."""
        formatted = []
        for chunk in evidence:
            page_info = f"Page: {chunk.page}" if chunk.page else "Page: Unknown"
            section_info = f", Section: {chunk.section}" if chunk.section else ""
            
            chunk_str = (
                f"[CHUNK_ID: {chunk.chunk_id}] ({page_info}{section_info})\n"
                f"{chunk.text}\n"
                f"---"
            )
            formatted.append(chunk_str)
        return "\n".join(formatted)

    def _extract_citations(self, text: str, evidence: List[EvidenceChunk]) -> List[Citation]:
        """
        Extract citations from the generated text and map them to evidence chunks.
        Very basic implementation based on the [Source: chunk_id...] format.
        """
        import re
        citations = []
        # Look for [Source: chunk_id, page N] or variations
        pattern = r"\[Source:\s*([a-zA-Z0-9\-]+)[^\]]*\]"
        matches = re.finditer(pattern, text)
        
        seen_chunks = set()
        for match in matches:
            chunk_id = match.group(1)
            if chunk_id in seen_chunks:
                continue
                
            # Find corresponding evidence chunk
            for chunk in evidence:
                # Fuzzy match chunk_id (LLMs sometimes truncate or alter it slightly)
                if chunk_id in chunk.chunk_id or chunk.chunk_id in chunk_id:
                    citations.append(Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        page=chunk.page,
                        excerpt=chunk.text[:100] + "..." # Storing a snippet
                    ))
                    seen_chunks.add(chunk_id)
                    break
                    
        return citations

    def _calculate_grounding_score(self, text: str, citations: List[Citation], evidence: List[EvidenceChunk]) -> float:
        """
        Heuristic score: proportion of sentences that have citations,
        capped at 1.0. A real implementation would use an NLI model.
        """
        if not evidence:
            return 0.0
        if "Information not found" in text:
            return 1.0 # Correctly identified missing info
            
        sentences = [s for s in text.split(".") if len(s.strip()) > 10]
        if not sentences:
            return 0.0
            
        # Simplistic: count explicit citation tags vs number of factual sentences
        citation_count = text.count("[Source:")
        
        score = citation_count / len(sentences) * 2.0 # Assume 1 citation per 2 sentences is excellent
        return min(1.0, float(score))
