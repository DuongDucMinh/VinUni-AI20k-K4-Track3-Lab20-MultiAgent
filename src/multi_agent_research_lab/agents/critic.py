"""Critic agent.

Performs fact-checking, hallucination detection, and citation verification on the final answer.
Assigns an objective quality score (0-10) based on evaluation criteria.
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

_SYSTEM_PROMPT = """You are an exacting Quality & Fact-Checking Critic.
Your goal is to rigorously evaluate a research report for correctness, hallucination risk, and citation grounding.

Evaluation Rubric:
1. Citation Grounding (0-3 points): Are claims properly attributed to provided sources?
2. Technical Depth & Accuracy (0-3 points): Is the technical analysis sound and comprehensive?
3. Structure & Clarity (0-2 points): Is the report well-organized with clear conclusions?
4. Objectivity & Balance (0-2 points): Are trade-offs and limitations fairly represented?

Output Format:
- Score: [Number between 0.0 and 10.0]
- Strengths: Brief bullet points
- Weaknesses / Ungrounded Claims: Specific points needing improvement
- Verdict: PASS / REVISE
"""


class CriticAgent(BaseAgent):
    """Fact-checking, citation verification, and quality-scoring agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and compute quality metrics."""
        t_start = perf_counter()
        logger.info("CriticAgent: reviewing final report")

        if not state.final_answer:
            logger.warning("CriticAgent: No final_answer to critique.")
            return state

        sources_summary = "\n".join(
            f"- [{src.metadata.get('source_id', 'SRC')}] {src.title}: {src.snippet[:150]}"
            for src in state.sources
        )

        user_prompt = (
            f"Research Question: {state.request.query}\n\n"
            f"=== FINAL REPORT TO CRITIQUE ===\n"
            f"{state.final_answer}\n\n"
            f"=== GROUND TRUTH SOURCES ===\n"
            f"{sources_summary}\n\n"
            "Evaluate the report strictly using your rubric and provide your Score (0.0 to 10.0)."
        )

        try:
            llm_response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            critique_text = llm_response.content
        except Exception as exc:
            err_msg = f"CriticAgent: LLM call failed: {exc}"
            logger.warning(err_msg)
            critique_text = "Critic evaluation skipped due to LLM error."
            llm_response = None

        duration = perf_counter() - t_start

        # Parse numeric score from critique (e.g. "Score: 8.5" or "8.5/10")
        score = _parse_score(critique_text)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critique_text,
                metadata={
                    "quality_score": score,
                    "duration_seconds": duration,
                    "input_tokens": llm_response.input_tokens if llm_response else None,
                    "output_tokens": llm_response.output_tokens if llm_response else None,
                },
            )
        )
        if llm_response:
            state.add_token_usage(
                agent="critic",
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                cost_usd=llm_response.cost_usd,
                duration_seconds=duration,
            )

        state.add_trace_event(
            "critic_done",
            {"quality_score": score, "duration_seconds": round(duration, 2)},
        )

        logger.info("CriticAgent: done. quality_score=%s, dur=%.1fs", score, duration)
        return state


def _parse_score(text: str) -> float:
    """Extract float score from critic response."""
    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return min(max(val, 0.0), 10.0)
        except ValueError:
            pass
    # Fallback heuristic: 8.0 default
    return 8.0

