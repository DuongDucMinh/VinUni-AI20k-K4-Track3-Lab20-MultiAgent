"""Tests for SupervisorAgent routing policy and state-machine transitions."""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_initially() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_RESEARCHER
    assert state.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Found notes on MAS architectures [SRC-01].",
        sources=[SourceDocument(title="Doc 1", snippet="Text")],
    )
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_ANALYST


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Notes",
        analysis_notes="Analysis",
    )
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_WRITER


def test_supervisor_routes_to_done_after_writer() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        research_notes="Notes",
        analysis_notes="Analysis",
        final_answer="Comprehensive final report.",
    )
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_DONE


def test_supervisor_enforces_max_iterations_guard() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=10,  # exceeds default max_iterations=6
    )
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_DONE
    assert any("max_iterations" in err for err in state.errors)

