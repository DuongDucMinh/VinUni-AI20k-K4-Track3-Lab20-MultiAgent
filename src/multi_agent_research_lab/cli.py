import io
import sys
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure Windows stdout supports UTF-8 characters without charmap errors
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import init_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console(safe_box=True, highlight=False)



def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline LLM call and display output and metrics."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    llm = LLMClient()
    system_prompt = (
        "You are an AI research assistant. Please research and write a structured, "
        "concise, and informative summary addressing the user's query."
    )

    console.print(f"[bold blue]Running Single-Agent Baseline for:[/bold blue] {query}")

    try:
        response = llm.complete(system_prompt=system_prompt, user_prompt=query)
        state.final_answer = response.content
        state.add_token_usage(
            agent="baseline_single_agent",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"model_used": response.model_used},
            )
        )
    except Exception as exc:
        console.print(Panel.fit(f"Baseline error: {exc}", title="Error", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Result", border_style="cyan"))

    table = Table(title="Baseline Execution Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model Used", str(response.model_used))
    table.add_row("Input Tokens", str(response.input_tokens))
    table.add_row("Output Tokens", str(response.output_tokens))
    table.add_row("Cost (USD)", f"${response.cost_usd:.6f}" if response.cost_usd else "$0.00")
    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic/--no-critic", help="Include critic agent")] = True,
) -> None:
    """Run the multi-agent workflow graph with Supervisor, Researcher, Analyst, Writer, and Critic."""
    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow(include_critic=critic)

    console.print(f"[bold green]Starting Multi-Agent Workflow for:[/bold green] {query}")

    result = workflow.run(state)

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Multi-Agent Final Report", border_style="green"))

    # Summary table
    table = Table(title="Multi-Agent Execution Summary")
    table.add_column("Agent / Metric", style="cyan")
    table.add_column("Details", style="magenta")

    table.add_row("Total Iterations", str(result.iteration))
    table.add_row("Route History", " -> ".join(result.route_history))
    table.add_row("Sources Retrieved", str(len(result.sources)))
    table.add_row("Citations Used", ", ".join(result.citation_ids_used) if result.citation_ids_used else "None")
    table.add_row("Input Tokens", str(result.token_usage.get("input", 0)))
    table.add_row("Output Tokens", str(result.token_usage.get("output", 0)))
    table.add_row("Total Cost (USD)", f"${result.total_cost_usd:.6f}")
    table.add_row("Errors Encountered", str(len(result.errors)))

    console.print(table)


if __name__ == "__main__":
    app()

