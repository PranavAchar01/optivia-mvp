"""
Agent 13B — Critical-Path Scheduler (§5.3.16).

Computes the critical path of the fleet DAG via the Critical Path Method
(CPM): forward pass for earliest start/finish, backward pass for latest
start/finish, slack = LS - ES, and zero-slack nodes form the critical path.
The longest weighted path length is normalized → cpl_norm, which feeds the
bandit's context vector (§5.3.12 Routing Context Vector x_t).
"""

from __future__ import annotations

from typing import Any

import structlog

from backend.pipeline.state import OptiviaState

log = structlog.get_logger(__name__)


def _topological_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn's algorithm. Assumes edges form a DAG (validator already ran)."""
    in_deg = {n: 0 for n in nodes}
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in in_deg and v in in_deg:
            in_deg[v] += 1
            succ[u].append(v)
    ready = [n for n, d in in_deg.items() if d == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in succ[n]:
            in_deg[m] -= 1
            if in_deg[m] == 0:
                ready.append(m)
    return order


def _cpm(
    nodes: list[str],
    edges: list[tuple[str, str]],
    duration: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return {node: {ES, EF, LS, LF, slack}}."""
    preds: dict[str, list[str]] = {n: [] for n in nodes}
    succs: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in preds and v in preds:
            preds[v].append(u)
            succs[u].append(v)

    order = _topological_order(nodes, edges)
    es: dict[str, float] = {}
    ef: dict[str, float] = {}
    for n in order:
        es[n] = max((ef[p] for p in preds[n] if p in ef), default=0.0)
        ef[n] = es[n] + duration.get(n, 0.0)

    ef_max = max(ef.values(), default=0.0)
    lf: dict[str, float] = {}
    ls: dict[str, float] = {}
    for n in reversed(order):
        lf[n] = min((ls[s] for s in succs[n] if s in ls), default=ef_max)
        ls[n] = lf[n] - duration.get(n, 0.0)

    return {
        n: {
            "ES": es.get(n, 0.0),
            "EF": ef.get(n, 0.0),
            "LS": ls.get(n, 0.0),
            "LF": lf.get(n, 0.0),
            "slack": ls.get(n, 0.0) - es.get(n, 0.0),
        }
        for n in nodes
    }


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(0.9 * (len(s) - 1)))
    return s[idx]


async def critical_path_scheduler(state: OptiviaState) -> OptiviaState:
    """Agent 13B — Critical-Path Scheduler."""
    fleet_dag = state.get("fleet_dag") or {}
    raw_nodes = fleet_dag.get("Nodes") or []
    raw_edges = fleet_dag.get("Edges") or []

    if not raw_nodes:
        state["critical_path"] = "No nodes generated."
        state["cpl_norm"] = 0.0  # type: ignore[typeddict-unknown-key]
        return state

    names: list[str] = [n.get("Name", f"node_{i}") for i, n in enumerate(raw_nodes)]
    duration: dict[str, float] = {
        n.get("Name", f"node_{i}"): float(n.get("Estimated Duration", 30.0))
        for i, n in enumerate(raw_nodes)
    }
    edges: list[tuple[str, str]] = [
        (e.get("From", ""), e.get("To", "")) for e in raw_edges
    ]

    metrics = _cpm(names, edges, duration)

    # Critical path = zero-slack chain through max-EF node, walked back via predecessors
    critical_set = {n for n, m in metrics.items() if abs(m["slack"]) < 1e-6}

    # Walk a single representative path for display
    ef_sorted = sorted(metrics.items(), key=lambda kv: kv[1]["EF"], reverse=True)
    path: list[str] = []
    if ef_sorted:
        cur = ef_sorted[0][0]
        path = [cur]
        preds: dict[str, list[str]] = {n: [] for n in names}
        for u, v in edges:
            if v in preds:
                preds[v].append(u)
        while preds.get(cur):
            cur = max(
                preds[cur],
                key=lambda p: metrics.get(p, {"EF": 0.0})["EF"],
            )
            path.append(cur)
        path.reverse()

    critical_path_length = max((m["EF"] for m in metrics.values()), default=0.0)
    total_duration = sum(duration.values())
    cpl_norm = (critical_path_length / total_duration) if total_duration > 0 else 0.0

    # Bottleneck = critical-path node whose duration ≥ p90 across the fleet
    p90 = _p90(list(duration.values()))
    bottlenecks = [n for n in critical_set if duration.get(n, 0.0) >= p90]

    # Annotate fleet_dag nodes with CPM verdicts so visualizer can highlight them
    for node in raw_nodes:
        name = node.get("Name")
        m = metrics.get(name, {})
        node["ES"] = round(m.get("ES", 0.0), 2)
        node["EF"] = round(m.get("EF", 0.0), 2)
        node["LS"] = round(m.get("LS", 0.0), 2)
        node["LF"] = round(m.get("LF", 0.0), 2)
        node["Slack"] = round(m.get("slack", 0.0), 2)
        node["On Critical Path"] = name in critical_set
        node["Bottleneck"] = name in bottlenecks

    fleet_dag["Nodes"] = raw_nodes
    fleet_dag["Critical Path"] = " -> ".join(path)
    fleet_dag["Critical Path Length"] = round(critical_path_length, 2)
    fleet_dag["Bottlenecks"] = bottlenecks
    state["fleet_dag"] = fleet_dag

    state["critical_path"] = " -> ".join(path)
    state["cpl_norm"] = round(cpl_norm, 4)  # type: ignore[typeddict-unknown-key]

    log.info(
        "critical_path_scheduler.done",
        n_nodes=len(names),
        cp_length=critical_path_length,
        cpl_norm=cpl_norm,
        bottlenecks=bottlenecks,
    )
    return state
