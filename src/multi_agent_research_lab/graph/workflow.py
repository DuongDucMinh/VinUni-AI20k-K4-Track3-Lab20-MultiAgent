"""LangGraph workflow for the multi-agent research system.

Constructs a cyclic StateGraph with the Supervisor routing tasks to
the Researcher, Analyst, Writer, and Critic agents until the final report is produced.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


def _route_decision(state: ResearchState) -> str:
    """Read the last route recorded by the Supervisor."""
    if not state.route_history:
        return ROUTE_RESEARCHER
    last_route = state.route_history[-1]
    logger.debug("Conditional routing decision: %s", last_route)
    return last_route


class MultiAgentWorkflow:
    """Builds and orchestrates the multi-agent graph."""

    def __init__(self, include_critic: bool = True) -> None:
        self.include_critic = include_critic
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent() if include_critic else None

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node_supervisor"):
            return self._supervisor.run(state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node_researcher"):
            return self._researcher.run(state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node_analyst"):
            return self._analyst.run(state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node_writer"):
            return self._writer.run(state)

    def _critic_node(self, state: ResearchState) -> ResearchState:
        with trace_span("node_critic"):
            if self._critic:
                return self._critic.run(state)
            return state

    def build(self) -> Any:
        """Create and compile the LangGraph StateGraph."""
        graph = StateGraph(ResearchState)

        # 1. Register all nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        if self.include_critic:
            graph.add_node("critic", self._critic_node)

        # 2. Set entry point
        graph.set_entry_point("supervisor")

        # 3. Add conditional edge from supervisor
        destinations = {
            ROUTE_RESEARCHER: "researcher",
            ROUTE_ANALYST: "analyst",
            ROUTE_WRITER: "writer",
            ROUTE_DONE: END,
        }
        if self.include_critic:
            destinations[ROUTE_CRITIC] = "critic"

        graph.add_conditional_edges("supervisor", _route_decision, destinations)

        # 4. Loop back to supervisor after each worker finishes
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        if self.include_critic:
            graph.add_edge("critic", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the multi-agent graph with end-to-end tracing."""
        with trace_span("multi_agent_workflow", {"query": state.request.query[:80]}):
            compiled = self.build()
            result = compiled.invoke(state)

            # Convert result back to ResearchState if necessary
            if isinstance(result, ResearchState):
                final_state = result
            elif isinstance(result, dict):
                final_state = ResearchState.model_validate(result)
            else:
                final_state = state

            logger.info(
                "MultiAgentWorkflow finished. Iterations: %d, Routes: %s, Cost: $%.5f",
                final_state.iteration,
                final_state.route_history,
                final_state.total_cost_usd,
            )
            return final_state

