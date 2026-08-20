"""Automated benchmark runner for Lab 20.

Runs Single-Agent Baseline and Multi-Agent Workflow on selected research queries,
collects metrics (latency, cost, quality, citation coverage, failure rate),
and renders a comprehensive markdown benchmark report to reports/benchmark_report.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import init_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
    "What are the trade-offs between single-agent and multi-agent architectures for complex research tasks?",
    "How does role specialization improve performance in multi-agent systems?",
    "What are the benefits and failure modes of reflection and self-correction loops in language agents?",
]


def single_agent_runner(query: str) -> ResearchState:
    """Execute single-agent baseline."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are an AI research assistant. Please research and write a structured, "
        "concise, and informative summary addressing the user's query."
    )
    resp = llm.complete(system_prompt=system_prompt, user_prompt=query)
    state.final_answer = resp.content
    state.add_token_usage(
        agent="single_agent_baseline",
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
    )
    return state


def multi_agent_runner(query: str) -> ResearchState:
    """Execute multi-agent workflow."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow(include_critic=True)
    return workflow.run(state)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_tracing()

    print("=" * 70)
    print("STARTING BENCHMARK: Single-Agent vs Multi-Agent System")
    print(f"Model: {settings.openai_model} | Base URL: {settings.openai_base_url or 'Groq Auto'}")
    print("=" * 70)

    all_metrics: list[BenchmarkMetrics] = []

    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n--- Task {i}/{len(BENCHMARK_QUERIES)}: {query[:60]}... ---")

        # 1. Single-Agent
        print("  -> Running Single-Agent Baseline...")
        _, m_single = run_benchmark(
            run_name=f"Single-Agent (Task {i})",
            query=query,
            runner=single_agent_runner,
        )
        all_metrics.append(m_single)
        print(f"     Done: {m_single.latency_seconds:.2f}s | Cost: ${m_single.estimated_cost_usd:.6f}")

        # 2. Multi-Agent
        print("  -> Running Multi-Agent Workflow...")
        _, m_multi = run_benchmark(
            run_name=f"Multi-Agent (Task {i})",
            query=query,
            runner=multi_agent_runner,
        )
        all_metrics.append(m_multi)
        print(f"     Done: {m_multi.latency_seconds:.2f}s | Cost: ${m_multi.estimated_cost_usd:.6f} | Quality: {m_multi.quality_score}/10 | Citations: {m_multi.citation_coverage:.0%}")

    # Generate Markdown Report
    report_content = render_markdown_report(all_metrics)
    report_path = Path("reports") / "benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"BENCHMARK FINISHED. Report written to {report_path.resolve()}")
    print("=" * 70)
    print("\n" + report_content)


if __name__ == "__main__":
    main()
