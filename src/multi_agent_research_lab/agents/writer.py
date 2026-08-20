"""Writer agent.

Produces the final comprehensive research report from research and analysis notes,
grounded in citations from the collected source documents.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a Principal Technical Writer and Research Synthesizer.
Your mission is to produce a high-impact, comprehensive, and well-structured final research report.

Report Structure Guidelines:
1. Executive Summary: High-level overview of the findings.
2. Background & Architecture: Core concepts, mechanisms, and taxonomy.
3. Comparative Evaluation & Trade-offs: Detailed breakdown with pros, cons, and performance trade-offs.
4. Production Engineering & Guardrails: Practical guidance, failure modes, and mitigation strategies.
5. Synthesis & Conclusions: Clear final verdict and recommendations.
6. References & Citations: List of all referenced sources with their [source_id] and brief descriptions.

Citation Rules:
- You MUST cite evidence using bracket notation like [SRC-01] or [KB-ART-01] for all key claims.
- Only reference sources provided in the context.
- Keep the writing polished, concise, authoritative, and accessible to technical practitioners.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` and extract `state.citation_ids_used`."""
        t_start = perf_counter()
        logger.info("WriterAgent: synthesizing final report")

        sources_summary = "\n".join(
            f"- [{src.metadata.get('source_id', 'SRC')}] {src.title}: {src.snippet[:200]}"
            for src in state.sources
        )

        user_prompt = (
            f"Research Question: {state.request.query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"=== RESEARCH NOTES ===\n"
            f"{state.research_notes or 'None'}\n\n"
            f"=== ANALYTICAL INSIGHTS ===\n"
            f"{state.analysis_notes or 'None'}\n\n"
            f"=== AVAILABLE SOURCE CITATIONS ===\n"
            f"{sources_summary or 'None'}\n\n"
            "Synthesize these notes into a complete, publication-ready research report with inline citations."
        )

        try:
            llm_response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            err_msg = f"WriterAgent: LLM call failed: {exc}"
            logger.error(err_msg)
            state.errors.append(err_msg)
            state.final_answer = (
                f"# Research Report: {state.request.query}\n\n"
                f"## Notes (Fallback)\n{state.research_notes or 'No notes available'}"
            )
            return state

        duration = perf_counter() - t_start
        final_text = llm_response.content

        # Extract citation references from the generated report
        cited_ids = _extract_citations(final_text, state.sources)
        state.citation_ids_used = cited_ids
        state.final_answer = final_text

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_text,
                metadata={
                    "citations_count": len(cited_ids),
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                    "cost_usd": llm_response.cost_usd,
                    "duration_seconds": duration,
                    "model_used": llm_response.model_used,
                },
            )
        )
        state.add_token_usage(
            agent="writer",
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
            duration_seconds=duration,
        )
        state.add_trace_event(
            "writer_done",
            {
                "report_length": len(final_text),
                "cited_ids": cited_ids,
                "tokens": (llm_response.input_tokens, llm_response.output_tokens),
                "duration_seconds": round(duration, 2),
                "model_used": llm_response.model_used,
            },
        )

        logger.info(
            "WriterAgent: done. length=%d chars, citations=%d, cost=$%.5f, dur=%.1fs",
            len(final_text),
            len(cited_ids),
            llm_response.cost_usd or 0,
            duration,
        )
        return state


def _extract_citations(text: str, sources: list) -> list[str]:
    """Find all source_ids referenced inside bracket notations in the text."""
    cited = set()
    # Match patterns like [SRC-01], [KB-01], [DOC-01], etc.
    bracket_tokens = re.findall(r"\[([A-Za-z0-9_\-]+)\]", text)
    available_sids = {src.metadata.get("source_id", "").lower(): src.metadata.get("source_id", "")
                      for src in sources if src.metadata.get("source_id")}

    for tok in bracket_tokens:
        tok_lower = tok.lower()
        if tok_lower in available_sids:
            cited.add(available_sids[tok_lower])
        else:
            # Also capture general matches
            cited.add(tok)

    return sorted(cited)

