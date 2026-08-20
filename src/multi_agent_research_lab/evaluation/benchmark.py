"""Benchmark runner for single-agent vs multi-agent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import save_trace_log

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute a runner, measure all benchmark metrics, and save execution trace."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Extract quality score from Critic result if available
    quality_score = 7.5  # default baseline score
    for res in state.agent_results:
        if res.agent == AgentName.CRITIC and "quality_score" in res.metadata:
            quality_score = float(res.metadata["quality_score"])
            break

    # Calculate citation coverage: cited sources / total available sources
    num_sources = len(state.sources)
    num_cited = len(state.citation_ids_used)
    citation_coverage = round(min(num_cited / max(num_sources, 1), 1.0), 2) if num_sources > 0 else 0.0

    # Calculate failure rate: accumulated errors / (total iterations + 1)
    failure_rate = round(min(len(state.errors) / max(state.iteration, 1), 1.0), 2)

    # Save structured trace
    trace_path = save_trace_log(run_name, state.trace)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=round(state.total_cost_usd, 6),
        quality_score=round(quality_score, 1),
        citation_coverage=citation_coverage,
        failure_rate=failure_rate,
        notes=f"iterations={state.iteration}, routes={state.route_history}, trace={trace_path.name}",
    )

    logger.info("Benchmark [%s] completed: latency=%.2fs, cost=$%.6f, quality=%.1f, citations=%.0f%%",
                run_name, latency, state.total_cost_usd, quality_score, citation_coverage * 100)

    return state, metrics

