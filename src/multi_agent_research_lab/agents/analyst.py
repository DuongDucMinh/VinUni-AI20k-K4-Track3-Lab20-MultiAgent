"""Analyst agent.

Turns research notes into structured analytical insights:
- Extracts core claims and arguments
- Compares viewpoints and flags contradictions
- Evaluates evidence reliability and gaps
"""

from __future__ import annotations

import logging
from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert Research Analyst.
Your role is to analyze research notes and produce structured analytical insights.

Structure your analysis into the following concise sections:
1. Key Findings & Core Claims: Summarize primary arguments with citation tags (e.g. [SRC-01]).
2. Comparative Analysis & Trade-offs: Contrast perspectives, methodologies, or architectural trade-offs.
3. Evidence Evaluation & Limitations: Identify strong vs weak evidence, unverified assumptions, or data gaps.
4. Strategic Implications & Recommendations: Actionable takeaways.

Rules:
- Be concise, objective, and analytical.
- Preserve source citation identifiers [source_id] wherever relevant.
- Highlight any unresolved conflicts with [CONFLICT].
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        t_start = perf_counter()
        logger.info("AnalystAgent: starting analysis")

        if not state.research_notes:
            err_msg = "AnalystAgent: research_notes is empty or missing."
            logger.warning(err_msg)
            state.errors.append(err_msg)
            state.analysis_notes = "No research notes available to analyze."
            return state

        user_prompt = (
            f"Research Query: {state.request.query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"=== RESEARCH NOTES ===\n"
            f"{state.research_notes}\n"
            f"=== END RESEARCH NOTES ===\n\n"
            "Produce structured analytical insights following your specified sections."
        )

        try:
            llm_response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            err_msg = f"AnalystAgent: LLM call failed: {exc}"
            logger.error(err_msg)
            state.errors.append(err_msg)
            state.analysis_notes = f"Analysis failed due to error: {exc}"
            return state

        duration = perf_counter() - t_start

        state.analysis_notes = llm_response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=llm_response.content,
                metadata={
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                    "cost_usd": llm_response.cost_usd,
                    "duration_seconds": duration,
                    "model_used": llm_response.model_used,
                },
            )
        )
        state.add_token_usage(
            agent="analyst",
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
            duration_seconds=duration,
        )
        state.add_trace_event(
            "analyst_done",
            {
                "analysis_length": len(llm_response.content),
                "tokens": (llm_response.input_tokens, llm_response.output_tokens),
                "duration_seconds": round(duration, 2),
                "model_used": llm_response.model_used,
            },
        )

        logger.info(
            "AnalystAgent: done. analysis=%d chars, cost=$%.5f, dur=%.1fs",
            len(llm_response.content),
            llm_response.cost_usd or 0,
            duration,
        )
        return state

