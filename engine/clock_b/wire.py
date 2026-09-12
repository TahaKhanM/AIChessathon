"""Install the clock_b allocator on a live ``engine.agent_rx.Agent``.

FIX-RXAGENT owns packaging/RSS; this race owns the clock. The one-line
wiring is ``attach(agent)``. ``engine.search`` is not modified.
"""

from __future__ import annotations

from engine.clock_b.allocator import SoftAllocator
from engine.clock_b.scaler import SoftScaler


def attach(agent, *, reserve_ms: float | None = None, overhead_ms: float = 25.0):
    """Replace the agent's allocator/scaler in place. Returns the agent."""
    kw = {}
    if reserve_ms is not None:
        kw["reserve_ms"] = reserve_ms
    agent.allocator = SoftAllocator(**kw)
    agent.scaler = SoftScaler()
    agent.overhead_ms = float(overhead_ms)
    return agent
