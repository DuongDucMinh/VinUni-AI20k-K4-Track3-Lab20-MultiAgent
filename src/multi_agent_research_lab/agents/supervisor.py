"""Supervisor / router agent.

Routes between worker agents (researcher → analyst → writer) based on
what is present in the shared state. Enforces max_iterations and failure fallback.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Route constants — these must match LangGraph node names
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (state machine):
        START            → researcher   (no research_notes yet)
        after researcher → analyst      (research_notes exists, no analysis_notes)
        after analyst    → writer       (analysis_notes exists, no final_answer)
        after writer     → done         (final_answer exists)
        any step         → done         (iteration >= max_iterations)
        repeated errors  → writer       (force finish rather than infinite retry)
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Determine the next route and record it in state."""
        settings = get_settings()
        max_iter = settings.max_iterations

        # ----- Guard: max iterations -----
        if state.iteration >= max_iter:
            logger.warning(
                "Supervisor: max_iterations=%d reached. Forcing route=done.", max_iter
            )
            state.record_route(ROUTE_DONE)
            state.errors.append(
                f"Supervisor forced termination after {state.iteration} iterations "
                f"(max_iterations={max_iter})."
            )
            state.add_trace_event(
                "supervisor_decision",
                {"route": ROUTE_DONE, "reason": "max_iterations_exceeded",
                 "iteration": state.iteration},
            )
            return state

        # ----- Guard: too many accumulated errors → skip to writer -----
        if len(state.errors) >= 3 and state.final_answer is None:
            logger.warning(
                "Supervisor: %d errors accumulated, routing to writer as fallback.",
                len(state.errors),
            )
            next_route = ROUTE_WRITER

        # ----- Normal routing -----
        elif state.research_notes is None:
            next_route = ROUTE_RESEARCHER
        elif state.analysis_notes is None:
            next_route = ROUTE_ANALYST
        elif state.final_answer is None:
            next_route = ROUTE_WRITER
        else:
            next_route = ROUTE_DONE

        reason = _routing_reason(state, next_route)
        logger.info(
            "Supervisor: iteration=%d route=%s reason=%s",
            state.iteration, next_route, reason,
        )

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_decision",
            {
                "route": next_route,
                "reason": reason,
                "iteration": state.iteration,
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
                "errors": len(state.errors),
            },
        )
        return state


def _routing_reason(state: ResearchState, route: str) -> str:
    if route == ROUTE_RESEARCHER:
        return "research_notes is None"
    if route == ROUTE_ANALYST:
        return "analysis_notes is None"
    if route == ROUTE_WRITER:
        return "final_answer is None (or error fallback)"
    if route == ROUTE_DONE:
        return "final_answer exists"
    return "unknown"

