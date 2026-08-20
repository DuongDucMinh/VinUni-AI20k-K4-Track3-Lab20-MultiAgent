"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    # Core request
    request: ResearchQuery

    # Workflow control
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    # Agent outputs
    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    # Agent execution records
    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # --- Benchmark / observability fields ---
    # Cumulative token usage {"input": N, "output": N}
    token_usage: dict[str, int] = Field(default_factory=lambda: {"input": 0, "output": 0})
    # Cumulative API cost in USD
    total_cost_usd: float = 0.0
    # Source IDs cited in the final answer (for citation_coverage metric)
    citation_ids_used: list[str] = Field(default_factory=list)
    # Per-agent wall-clock durations in seconds
    agent_durations: dict[str, float] = Field(default_factory=dict)

    # ------------------------------------------------------------------ helpers

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_token_usage(
        self,
        agent: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
        duration_seconds: float | None = None,
    ) -> None:
        """Accumulate token counts, cost, and duration from one LLM call."""
        if input_tokens is not None:
            self.token_usage["input"] = self.token_usage.get("input", 0) + input_tokens
        if output_tokens is not None:
            self.token_usage["output"] = self.token_usage.get("output", 0) + output_tokens
        if cost_usd is not None:
            self.total_cost_usd += cost_usd
        if duration_seconds is not None:
            # Accumulate in case the same agent is called multiple times
            self.agent_durations[agent] = (
                self.agent_durations.get(agent, 0.0) + duration_seconds
            )

