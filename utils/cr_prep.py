"""CommonRoad-scenario pre-processing for the Lanelet2 -> OpenDRIVE direction.

crdesigner's Lanelet2 reader produces a ``Scenario`` whose lanelets are 3-D (``ele`` tags),
sparsely sampled (hand-drawn Autoware maps: median 4 nodes per boundary) and whose
adjacency may be inconsistent. Its OpenDRIVE writer (``cr2odr``) assumes 2-D, roughly
equidistant samples and acyclic, reciprocal adjacency. The functions here bridge that gap
and leave the scenario otherwise untouched. Each returns a stats dict for the CLI.

All functions rebuild ``Lanelet`` objects instead of mutating vertex arrays in place: the
commonroad ``Lanelet`` caches derived data (polygon, distances) and ``LaneletNetwork`` keeps
an STRtree built from the original geometry.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from commonroad.geometry.polyline_util import resample_polyline_with_number
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.scenario import Scenario

OVERLAP_TOLERANCE_M = 0.05


# --- helpers -----------------------------------------------------------------------------

def _rebuild(lanelet: Lanelet, left: np.ndarray, center: np.ndarray, right: np.ndarray) -> Lanelet:
    """Copy of ``lanelet`` with new vertex arrays and all other attributes preserved."""
    return Lanelet(
        left_vertices=np.asarray(left, dtype=float),
        center_vertices=np.asarray(center, dtype=float),
        right_vertices=np.asarray(right, dtype=float),
        lanelet_id=lanelet.lanelet_id,
        predecessor=list(lanelet.predecessor),
        successor=list(lanelet.successor),
        adjacent_left=lanelet.adj_left,
        adjacent_left_same_direction=lanelet.adj_left_same_direction,
        adjacent_right=lanelet.adj_right,
        adjacent_right_same_direction=lanelet.adj_right_same_direction,
        line_marking_left_vertices=lanelet.line_marking_left_vertices,
        line_marking_right_vertices=lanelet.line_marking_right_vertices,
        stop_line=lanelet.stop_line,
        lanelet_type=lanelet.lanelet_type,
        user_one_way=lanelet.user_one_way,
        user_bidirectional=lanelet.user_bidirectional,
        traffic_signs=lanelet.traffic_signs,
        traffic_lights=lanelet.traffic_lights,
    )


def _replace_lanelets(scenario: Scenario, lanelets: list[Lanelet]) -> LaneletNetwork:
    """Swap the scenario's lanelet network for one built from ``lanelets``.

    Traffic signs/lights/areas/intersections and the osm-id bookkeeping
    (``_old_lanelet_ids``, set by crdesigner) are carried over.
    """
    old = scenario.lanelet_network
    new = LaneletNetwork()
    for la in lanelets:
        new.add_lanelet(la)
    for sign in old.traffic_signs:
        new.add_traffic_sign(sign, set())
    for light in old.traffic_lights:
        new.add_traffic_light(light, set())
    for area in getattr(old, "areas", []):
        new.add_area(area, set())
    for inter in old.intersections:
        new.add_intersection(inter)
    if hasattr(old, "_old_lanelet_ids"):
        new._old_lanelet_ids = dict(old._old_lanelet_ids)  # noqa: SLF001 (crdesigner bookkeeping)
    scenario.replace_lanelet_network(new)
    return new


def _polyline_length(poly: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(poly[:, :2], axis=0).T))) if len(poly) > 1 else 0.0


def _heading(lanelet: Lanelet) -> np.ndarray:
    c = lanelet.center_vertices[:, :2]
    v = c[-1] - c[0]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _coincide(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """True if every vertex of ``b`` lies within ``tol`` of polyline ``a`` (2-D)."""
    a = a[:, :2]
    seg = np.diff(a, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    for p in b[:, :2]:
        d_end = min(np.linalg.norm(p - a[0]), np.linalg.norm(p - a[-1]))
        if d_end <= tol:
            continue
        rel = p - a[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, seg) / np.maximum(seg_len ** 2, 1e-12), 0, 1)
        proj = a[:-1] + seg * t[:, None]
        if np.min(np.hypot(*(p - proj).T)) > tol:
            return False
    return True


def _is_overlapping_pair(a: Lanelet, b: Lanelet, tol: float) -> bool:
    """Same strip travelled in opposite directions (Autoware two-way single lane)."""
    return _coincide(a.left_vertices, b.right_vertices[::-1], tol) and _coincide(
        a.right_vertices, b.left_vertices[::-1], tol
    )


# --- stages ------------------------------------------------------------------------------

def strip_z(scenario: Scenario) -> tuple[dict, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Drop the elevation column from every lanelet; return it keyed by lanelet id.

    The z profile is kept so a later stage can emit an OpenDRIVE ``elevationProfile``.
    """
    z_table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rebuilt = []
    n3d = 0
    for la in scenario.lanelet_network.lanelets:
        if la.left_vertices.shape[1] > 2 or la.right_vertices.shape[1] > 2:
            n3d += 1
            z_table[la.lanelet_id] = (
                la.left_vertices[:, 2].copy() if la.left_vertices.shape[1] > 2 else np.zeros(len(la.left_vertices)),
                la.right_vertices[:, 2].copy() if la.right_vertices.shape[1] > 2 else np.zeros(len(la.right_vertices)),
            )
            rebuilt.append(_rebuild(la, la.left_vertices[:, :2], la.center_vertices[:, :2], la.right_vertices[:, :2]))
        else:
            rebuilt.append(la)
    if n3d:
        _replace_lanelets(scenario, rebuilt)
    return {"lanelets": len(rebuilt), "lanelets_3d": n3d}, z_table


def sanitize_adjacency(scenario: Scenario, tol: float = OVERLAP_TOLERANCE_M) -> dict:
    """Make lanelet adjacency safe for cr2odr's road builder.

    Rules, in order:
      1. overlapping opposite-direction pairs are never adjacent (clear both sides);
      2. a *same-direction* link between lanelets with opposing headings is dropped;
      3. links are made reciprocal (fill the missing side; on conflict keep the existing
         link of the neighbour and drop ours);
      4. any cycle along same-direction ``adj_right`` / ``adj_left`` chains is cut;
      5. dangling or self references are removed.
    """
    net = scenario.lanelet_network
    byid = {la.lanelet_id: la for la in net.lanelets}
    stats = {"overlap_pairs": 0, "dropped_direction": 0, "made_reciprocal": 0,
             "conflicts_dropped": 0, "cycles_cut": 0, "dangling_cleared": 0}

    def clear(a: Lanelet, side: str) -> None:
        setattr(a, f"adj_{side}", None)
        setattr(a, f"adj_{side}_same_direction", None)

    def link(a: Lanelet, side: str, b_id: int, same: bool) -> None:
        setattr(a, f"adj_{side}", b_id)
        setattr(a, f"adj_{side}_same_direction", bool(same))

    # 5a. dangling / self
    for la in byid.values():
        for side in ("left", "right"):
            t = getattr(la, f"adj_{side}")
            if t is not None and (t == la.lanelet_id or t not in byid):
                clear(la, side)
                stats["dangling_cleared"] += 1

    # 1. overlapping pairs + 2. direction consistency
    seen_pairs: set[tuple[int, int]] = set()
    for la in byid.values():
        for side in ("left", "right"):
            t = getattr(la, f"adj_{side}")
            if t is None:
                continue
            nb = byid[t]
            same = getattr(la, f"adj_{side}_same_direction")
            key = (min(la.lanelet_id, t), max(la.lanelet_id, t))
            if _is_overlapping_pair(la, nb, tol):
                for s in ("left", "right"):
                    if getattr(la, f"adj_{s}") == t:
                        clear(la, s)
                    if getattr(nb, f"adj_{s}") == la.lanelet_id:
                        clear(nb, s)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    stats["overlap_pairs"] += 1
                continue
            if same and float(np.dot(_heading(la), _heading(nb))) < 0:
                clear(la, side)
                stats["dropped_direction"] += 1

    # 3. reciprocity
    for la in byid.values():
        for side in ("left", "right"):
            t = getattr(la, f"adj_{side}")
            if t is None:
                continue
            nb = byid[t]
            same = bool(getattr(la, f"adj_{side}_same_direction"))
            other = ("right" if side == "left" else "left") if same else side
            back = getattr(nb, f"adj_{other}")
            if back == la.lanelet_id:
                continue
            if back is None:
                link(nb, other, la.lanelet_id, same)
                stats["made_reciprocal"] += 1
            else:
                clear(la, side)
                stats["conflicts_dropped"] += 1

    # 4. cycles along same-direction chains
    for side in ("right", "left"):
        for start in byid.values():
            seen = [start.lanelet_id]
            cur = start
            while True:
                t = getattr(cur, f"adj_{side}")
                if t is None or not getattr(cur, f"adj_{side}_same_direction"):
                    break
                if t in seen:
                    other = "left" if side == "right" else "right"
                    nb = byid[t]
                    clear(cur, side)
                    if getattr(nb, f"adj_{other}") == cur.lanelet_id:
                        clear(nb, other)
                    stats["cycles_cut"] += 1
                    break
                seen.append(t)
                cur = byid[t]

    # commonroad's same_direction setter rejects None: normalise cleared flags to False
    for la in byid.values():
        for side in ("left", "right"):
            if getattr(la, f"adj_{side}") is None:
                setattr(la, f"adj_{side}_same_direction", False)
    return stats


def resample_boundaries(scenario: Scenario, step: float = 1.0, min_points: int = 3) -> dict:
    """Resample left/right boundaries by arc length to a common, roughly ``step``-spaced grid.

    cr2odr derives heading and curvature with equidistant-sample assumptions and fits
    line/arc/spiral primitives between consecutive samples; the sparse hand-placed nodes of
    Autoware maps make that fit erratic. End points are preserved exactly, so lanelet
    connectivity (resolved by id in CommonRoad) is unaffected.
    """
    rebuilt = []
    before = after = 0
    changed = 0
    for la in scenario.lanelet_network.lanelets:
        left, right = la.left_vertices[:, :2], la.right_vertices[:, :2]
        n_target = max(min_points, int(math.ceil(max(_polyline_length(left), _polyline_length(right)) / step)) + 1)
        before += len(left) + len(right)
        if n_target == len(left) == len(right):
            rebuilt.append(la)
            after += len(left) + len(right)
            continue
        new_left = resample_polyline_with_number(left, n_target)
        new_right = resample_polyline_with_number(right, n_target)
        new_left[0], new_left[-1] = left[0], left[-1]
        new_right[0], new_right[-1] = right[0], right[-1]
        rebuilt.append(_rebuild(la, new_left, (new_left + new_right) / 2.0, new_right))
        after += 2 * n_target
        changed += 1
    if changed:
        _replace_lanelets(scenario, rebuilt)
    return {"lanelets_resampled": changed, "vertices_before": before, "vertices_after": after}


def topology_report(scenario: Scenario) -> dict:
    net = scenario.lanelet_network
    ids = {la.lanelet_id for la in net.lanelets}
    succ = sum(len(la.successor) for la in net.lanelets)
    pred = sum(len(la.predecessor) for la in net.lanelets)
    dangling = sum(1 for la in net.lanelets for t in (*la.successor, *la.predecessor) if t not in ids)
    isolated = sum(1 for la in net.lanelets if not la.successor and not la.predecessor
                   and la.adj_left is None and la.adj_right is None)
    adj_same = sum(1 for la in net.lanelets for s in ("left", "right")
                   if getattr(la, f"adj_{s}") is not None and getattr(la, f"adj_{s}_same_direction"))
    adj_opp = sum(1 for la in net.lanelets for s in ("left", "right")
                  if getattr(la, f"adj_{s}") is not None and not getattr(la, f"adj_{s}_same_direction"))
    return {"lanelets": len(ids), "successor_links": succ, "predecessor_links": pred,
            "adjacent_same_direction": adj_same, "adjacent_opposite_direction": adj_opp,
            "dangling_links": dangling, "isolated_lanelets": isolated}
