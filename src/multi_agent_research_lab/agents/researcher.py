"""Researcher agent.

Collects sources from the offline corpus (and optionally live web), then uses
an LLM to synthesize concise research notes with explicit citation IDs.
"""

from __future__ import annotations

import logging
from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a meticulous Research Agent. Your job is to synthesize
information from provided source documents into clear, well-structured research notes.

Rules:
1. Only state facts supported by the provided sources.
2. After each key claim, add the source_id in brackets, e.g. [SRC-01].
3. Group related findings into short paragraphs with clear topic sentences.
4. Flag any conflicting information between sources with "CONFLICT:".
5. Do NOT fabricate facts or invent source IDs.
6. End with a "Key Sources Used:" section listing all source_ids you cited.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes with citations."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        t_start = perf_counter()
        logger.info("ResearcherAgent: starting search for '%s'", state.request.query[:80])

        # --- 1. Search for sources ---
        try:
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
        except Exception as exc:
            err_msg = f"ResearcherAgent: search failed: {exc}"
            logger.error(err_msg)
            state.errors.append(err_msg)
            sources = []

        state.sources = sources
        logger.info("ResearcherAgent: found %d sources", len(sources))

        if not sources:
            state.research_notes = (
                "No sources found. The researcher could not retrieve relevant documents "
                f"for the query: {state.request.query}"
            )
            state.add_trace_event("researcher_done", {"sources": 0, "warning": "no_sources"})
            return state

        # --- 2. Build source context for LLM ---
        source_context = _format_sources(sources)

        user_prompt = (
            f"Research Query: {state.request.query}\n\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"=== SOURCE DOCUMENTS ===\n{source_context}\n"
            f"=== END SOURCES ===\n\n"
            "Please write comprehensive research notes based ONLY on the above sources. "
            "Include citation IDs [source_id] for every key claim."
        )

        # --- 3. Call LLM ---
        try:
            llm_response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            err_msg = f"ResearcherAgent: LLM call failed: {exc}"
            logger.error(err_msg)
            state.errors.append(err_msg)
            # Fallback: use raw snippets as research notes
            state.research_notes = _fallback_notes(sources, state.request.query)
            return state

        duration = perf_counter() - t_start

        # --- 4. Update state ---
        state.research_notes = llm_response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=llm_response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                    "cost_usd": llm_response.cost_usd,
                    "duration_seconds": duration,
                },
            )
        )
        state.add_token_usage(
            agent="researcher",
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
            duration_seconds=duration,
        )
        state.add_trace_event(
            "researcher_done",
            {
                "sources": len(sources),
                "notes_length": len(llm_response.content),
                "tokens": (llm_response.input_tokens, llm_response.output_tokens),
                "duration_seconds": round(duration, 2),
            },
        )

        logger.info(
            "ResearcherAgent: done. notes=%d chars, cost=$%.5f, dur=%.1fs",
            len(llm_response.content),
            llm_response.cost_usd or 0,
            duration,
        )
        return state


def _format_sources(sources: list) -> str:
    """Format source documents into a readable block for the LLM prompt."""
    parts = []
    for i, src in enumerate(sources, 1):
        sid = src.metadata.get("source_id", f"SRC-{i:02d}")
        url_line = f"URL: {src.url}" if src.url else ""
        parts.append(
            f"[{sid}] {src.title}\n"
            f"{url_line}\n"
            f"{src.snippet}\n"
        )
    return "\n---\n".join(parts)


def _fallback_notes(sources: list, query: str) -> str:
    """Minimal fallback when LLM is unavailable."""
    lines = [f"Research notes for: {query}\n(LLM unavailable — raw snippets only)\n"]
    for src in sources:
        sid = src.metadata.get("source_id", "?")
        lines.append(f"[{sid}] {src.title}: {src.snippet[:300]}")
    return "\n\n".join(lines)

