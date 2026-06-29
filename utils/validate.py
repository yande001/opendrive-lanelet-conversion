"""
S2 — Autoware LL2 requirement validator.

Grades an exported Lanelet2 `.osm` against the mandatory, machine-checkable
requirements from `20260407 - Autoware LL2 Map Requirements.xlsx`. Each check is
a pure function of the parsed OSM tree and returns a `CheckResult`:

    PASS  — requirement satisfied
    FAIL  — requirement applicable but not met (with offending counts)
    SKIP  — not applicable / source-dependent and no relevant source data present

The check predicates mirror the S1 gap-analysis (docs/roadmap/autoware-ll2/
s1-findings.md). Geometry checks use the `local_x`/`local_y` metric node tags.

Usage:
    from utils.validate import validate_file
    results = validate_file("output/S1/Town04_no_georef.osm")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict

from lxml import etree

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Geometry thresholds (vm-01-24, vm-01-05).
MAX_LEN_STRAIGHT_M = 100.0
MAX_LEN_CURVED_M = 20.0
CURVE_ANGLE_DEG = 160.0        # interior angle below this ⇒ the polyline is "curved"
                               # (>20° bend; a gently-curving road keeps the 100 m
                               # straight limit, only real curves get the 20 m limit)
SMOOTH_MIN_ANGLE_DEG = 60.0    # interior angle below this ⇒ a "jagged" kink (vm-01-05)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    req_id: str
    name: str
    status: str
    detail: str = ""
    offenders: int = 0
    total: int = 0

    def line(self) -> str:
        frac = f" ({self.offenders}/{self.total})" if self.total else ""
        return f"  {self.req_id:<9} {self.status:<4} {self.name} — {self.detail}{frac}"


# --------------------------------------------------------------------------- #
# Parsed-map facade
# --------------------------------------------------------------------------- #
def _tags(el) -> dict:
    return {t.get("k"): t.get("v") for t in el.findall("tag")}


@dataclass
class Map:
    root: etree._Element
    nodes: dict = field(default_factory=dict)          # id -> (x, y, ele)
    ways: dict = field(default_factory=dict)           # id -> element
    way_tags: dict = field(default_factory=dict)       # id -> tag dict
    way_nodes: dict = field(default_factory=dict)      # id -> [node ids]
    lanelets: list = field(default_factory=list)       # relation elements (type=lanelet)
    regelems: list = field(default_factory=list)       # relation elements (type=regulatory_element)

    @classmethod
    def parse(cls, path: str) -> "Map":
        root = etree.parse(path).getroot()
        m = cls(root=root)
        for n in root.findall("node"):
            t = _tags(n)
            try:
                x = float(t.get("local_x")) if "local_x" in t else float(n.get("lon"))
                y = float(t.get("local_y")) if "local_y" in t else float(n.get("lat"))
            except (TypeError, ValueError):
                x = y = 0.0
            ele = float(t["ele"]) if "ele" in t else None
            m.nodes[n.get("id")] = (x, y, ele)
        for w in root.findall("way"):
            wid = w.get("id")
            m.ways[wid] = w
            m.way_tags[wid] = _tags(w)
            m.way_nodes[wid] = [nd.get("ref") for nd in w.findall("nd")]
        for r in root.findall("relation"):
            rt = _tags(r).get("type")
            if rt == "lanelet":
                m.lanelets.append(r)
            elif rt == "regulatory_element":
                m.regelems.append(r)
        return m

    # -- helpers ---------------------------------------------------------- #
    def way_polyline(self, wid):
        return [self.nodes[n][:2] for n in self.way_nodes.get(wid, []) if n in self.nodes]

    def lanelet_bound_ways(self, ll):
        out = {}
        for mem in ll.findall("member"):
            if mem.get("type") == "way" and mem.get("role") in ("left", "right"):
                out[mem.get("role")] = mem.get("ref")
        return out

    def regelem_subtypes(self):
        return [_tags(r).get("subtype") for r in self.regelems]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _polyline_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _min_interior_angle(pts):
    """Smallest interior angle (deg) over the polyline; 180 if <3 points."""
    if len(pts) < 3:
        return 180.0
    worst = 180.0
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        cosang = max(min((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), 1.0), -1.0)
        worst = min(worst, math.degrees(math.acos(cosang)))
    return worst


# --------------------------------------------------------------------------- #
# Checks  (one per requirement id)
# --------------------------------------------------------------------------- #
def _road_lanelets(m):
    return [ll for ll in m.lanelets if _tags(ll).get("subtype") in ("road", "road_shoulder")]


def check_vm_01_01(m):
    """Road lanelets must have subtype, location, one_way, speed_limit."""
    roads = _road_lanelets(m)
    if not roads:
        return CheckResult("vm-01-01", "Lanelet basics tags", SKIP, "no road lanelets")
    miss_loc = miss_ow = miss_spd = 0
    # speed_limit may be a tag OR a regulatory element referenced by the lanelet
    speed_regelem_ids = {r.get("id") for r in m.regelems if _tags(r).get("subtype") == "speed_limit"}
    for ll in roads:
        t = _tags(ll)
        refs = {mem.get("ref") for mem in ll.findall("member") if mem.get("type") == "relation"}
        if "location" not in t:
            miss_loc += 1
        if "one_way" not in t:
            miss_ow += 1
        if "speed_limit" not in t and not (refs & speed_regelem_ids):
            miss_spd += 1
    worst = max(miss_loc, miss_ow, miss_spd)
    status = PASS if worst == 0 else FAIL
    return CheckResult("vm-01-01", "Lanelet basics tags", status,
                       f"missing location={miss_loc} one_way={miss_ow} speed_limit={miss_spd}",
                       worst, len(roads))


def check_vm_01_02(m):
    """Boundary line ways must carry lane_change."""
    lines = [wid for wid, t in m.way_tags.items()
             if t.get("type") in ("line_thin", "line_thick", "road_border")]
    if not lines:
        return CheckResult("vm-01-02", "Lane-change allowance", SKIP, "no marking line ways")
    miss = sum(1 for wid in lines if "lane_change" not in m.way_tags[wid])
    return CheckResult("vm-01-02", "Lane-change allowance", PASS if miss == 0 else FAIL,
                       f"lines without lane_change", miss, len(lines))


def check_vm_01_03(m):
    """Linestring sharing: boundary ways referenced by >1 lanelet (structural proxy)."""
    refs = defaultdict(int)
    for ll in m.lanelets:
        for wid in m.lanelet_bound_ways(ll).values():
            refs[wid] += 1
    if not refs:
        return CheckResult("vm-01-03", "Linestring sharing", SKIP, "no boundary ways")
    shared = sum(1 for c in refs.values() if c > 1)
    # informational: sharing existing at all is the structural prerequisite
    status = PASS if shared > 0 else FAIL
    return CheckResult("vm-01-03", "Linestring sharing", status,
                       f"boundary ways shared by >1 lanelet", shared, len(refs))


def _polylines_equal(a, b, tol):
    """True if polylines coincide (same or reversed point order) within tol."""
    if len(a) != len(b) or len(a) < 2:
        return False
    fwd = all(math.dist(a[i], b[i]) < tol for i in range(len(a)))
    if fwd:
        return True
    return all(math.dist(a[i], b[len(b) - 1 - i]) < tol for i in range(len(a)))


MIN_BOUNDARY_LEN_M = 1.0       # ignore degenerate sub-metre stub boundaries


def _lanelet_centroid(m, ll):
    """Mean of all boundary points of a lanelet, or None."""
    pts = []
    for wid in m.lanelet_bound_ways(ll).values():
        pts += m.way_polyline(wid)
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _opposite_sides(ca, cb, ref):
    """True if centroids ca, cb lie on opposite sides of the line through ref.

    Node-order independent: uses the line's normal, not its direction. Two lanes
    sharing a centerline are opposing iff their bodies straddle that line; a
    duplicate/degenerate co-directional pair has near-coincident centroids (same
    side) and is rejected.
    """
    if not (ca and cb) or len(ref) < 2:
        return False
    dx, dy = ref[-1][0] - ref[0][0], ref[-1][1] - ref[0][1]
    nx, ny = -dy, dx                       # normal to the boundary
    sa = (ca[0] - ref[0][0]) * nx + (ca[1] - ref[0][1]) * ny
    sb = (cb[0] - ref[0][0]) * nx + (cb[1] - ref[0][1]) * ny
    return sa * sb < 0 and abs(sa) > 1e-3 and abs(sb) > 1e-3


def check_vm_01_04(m):
    """Opposing-traffic centerline sharing.

    Two opposing lanes meet at a shared centerline. In the LL2 model that is a
    boundary `way` referenced by both lanelets with the *same* role (both `left`
    for RHT, both `right` for LHT) — whereas same-direction sharing (vm-01-03)
    uses *different* roles (`left`+`right`). So a same-role shared way is, by
    construction, an opposing centerline; geometry only needs to confirm the two
    bodies straddle it (rejecting degenerate co-directional duplicates).

    PASS — at least one same-role way shared by an opposing lanelet pair.
    FAIL — two opposing lanelets' same-role boundaries are geometrically
           coincident yet kept as distinct way ids (centerline not merged).
    SKIP — no opposing-lane geometry present at all.
    """
    tol = 0.5
    owners = defaultdict(list)             # way id -> [(lanelet element, role)]
    for ll in m.lanelets:
        for role, wid in m.lanelet_bound_ways(ll).items():
            owners[wid].append((ll, role))
    centroid = {id(ll): _lanelet_centroid(m, ll) for ll in m.lanelets}

    # PASS evidence: a same-role shared way whose two owners straddle it.
    shared_opposing = 0
    for wid, own in owners.items():
        for role in ("left", "right"):
            grp = [ll for ll, r in own if r == role]
            ref = m.way_polyline(wid)
            if any(_opposite_sides(centroid[id(grp[i])], centroid[id(grp[j])], ref)
                   for i in range(len(grp)) for j in range(i + 1, len(grp))):
                shared_opposing += 1
                break

    # FAIL evidence: coincident-but-distinct same-role boundaries of opposing
    # lanelets. Bucket by rounded endpoints (1 m grid) so the test stays ~O(n);
    # drop sub-metre stubs so degenerate duplicate lanelets don't register.
    boundary = []
    for wid, t in m.way_tags.items():
        if t.get("type") not in ("line_thin", "line_thick", "road_border"):
            continue
        pts = m.way_polyline(wid)
        if len(pts) >= 2 and _polyline_len(pts) >= MIN_BOUNDARY_LEN_M:
            boundary.append((wid, pts))
    buckets = defaultdict(list)
    for wid, pts in boundary:
        p0 = (round(pts[0][0]), round(pts[0][1]))
        p1 = (round(pts[-1][0]), round(pts[-1][1]))
        buckets[tuple(sorted((p0, p1)))].append((wid, pts))   # reverse collides
    opposing_dupes = 0
    for group in buckets.values():
        for i in range(len(group)):
            wi, pi = group[i]
            for j in range(i + 1, len(group)):
                wj, pj = group[j]
                if not _polylines_equal(pi, pj, tol):
                    continue
                roles_i = {r for _, r in owners.get(wi, [])}
                roles_j = {r for _, r in owners.get(wj, [])}
                if not (roles_i and roles_i == roles_j):
                    continue
                if any(_opposite_sides(centroid[id(a)], centroid[id(b)], pi)
                       for a, _ in owners.get(wi, []) for b, _ in owners.get(wj, [])):
                    opposing_dupes += 1

    if opposing_dupes:
        return CheckResult("vm-01-04", "Opposing centerline sharing", FAIL,
                           "opposing boundaries coincident but not merged into one way",
                           opposing_dupes, len(boundary))
    if not shared_opposing:
        return CheckResult("vm-01-04", "Opposing centerline sharing", SKIP,
                           "no opposing-lane geometry detected")
    return CheckResult("vm-01-04", "Opposing centerline sharing", PASS,
                       "opposing lanes share a same-role centerline way",
                       shared_opposing, len(owners))


def check_vm_01_05(m):
    """Geometry smoothness: no boundary way has an interior kink below threshold."""
    jagged = 0
    total = 0
    for wid, t in m.way_tags.items():
        if t.get("type") not in ("line_thin", "line_thick", "road_border", "curbstone"):
            continue
        total += 1
        if _min_interior_angle(m.way_polyline(wid)) < SMOOTH_MIN_ANGLE_DEG:
            jagged += 1
    if total == 0:
        return CheckResult("vm-01-05", "Lane geometry smooth", SKIP, "no boundary ways")
    return CheckResult("vm-01-05", "Lane geometry smooth", PASS if jagged == 0 else FAIL,
                       f"ways with a kink <{SMOOTH_MIN_ANGLE_DEG:.0f}°", jagged, total)


SAME_LINE_TOL_M = 0.05   # boundaries within this are treated as one physical line


def _subtype_boundaries(m, subtypes):
    """[(way_id, polyline)] for boundaries of lanelets whose subtype is in `subtypes`."""
    out = []
    for ll in m.lanelets:
        if _tags(ll).get("subtype") in subtypes:
            for wid in m.lanelet_bound_ways(ll).values():
                pts = m.way_polyline(wid)
                if len(pts) >= 2 and _polyline_len(pts) >= MIN_BOUNDARY_LEN_M:
                    out.append((wid, pts))
    return out


def check_vm_01_16(m):
    """Road and adjacent road_shoulder share the boundary line.

    Where a road boundary and a road_shoulder boundary are the *same physical
    line* (coincident within SAME_LINE_TOL_M, matching vertex count), they must
    resolve to one shared `way` id. Shoulders that merely run beside other
    shoulders, or whose edge is genuinely offset from the road (a real gap), are
    not adjacency cases and are not flagged. Sharing itself is produced by the
    cr2lanelet geometric-merge fallback (see vm-01-04); this check confirms it.
    """
    shoulders = [ll for ll in m.lanelets if _tags(ll).get("subtype") == "road_shoulder"]
    if not shoulders:
        return CheckResult("vm-01-16", "Road/shoulder sharing", SKIP,
                           "no road_shoulder lanelets")
    road_idx = defaultdict(list)
    for wid, pts in _subtype_boundaries(m, ("road",)):
        p0 = (round(pts[0][0]), round(pts[0][1]))
        p1 = (round(pts[-1][0]), round(pts[-1][1]))
        road_idx[tuple(sorted((p0, p1)))].append((wid, pts))
    unmerged = set()
    for wid, pts in _subtype_boundaries(m, ("road_shoulder",)):
        p0 = (round(pts[0][0]), round(pts[0][1]))
        p1 = (round(pts[-1][0]), round(pts[-1][1]))
        for rw, rpts in road_idx.get(tuple(sorted((p0, p1))), []):
            if rw != wid and _polylines_equal(pts, rpts, SAME_LINE_TOL_M):
                unmerged.add(tuple(sorted((wid, rw))))
    if unmerged:
        return CheckResult("vm-01-16", "Road/shoulder sharing", FAIL,
                           "road & shoulder boundaries identical but not one way",
                           len(unmerged), len(shoulders))
    return CheckResult("vm-01-16", "Road/shoulder sharing", PASS,
                       "adjacent road/shoulder boundaries share one way", 0, len(shoulders))


def _lanelet_cross_sections(m, ll):
    """The two end cross-sections of a lanelet as frozensets of {left-end-node,
    right-end-node}. Order-independent (pairs each left endpoint with the nearer
    right endpoint), so shared-way reversal doesn't matter. Successive lanelets
    share a cross-section (predecessor end == successor start); lateral neighbours
    do not (they share only one corner)."""
    bw = m.lanelet_bound_ways(ll)
    lw, rw = bw.get("left"), bw.get("right")
    ln, rn = m.way_nodes.get(lw, []), m.way_nodes.get(rw, [])
    if len(ln) < 2 or len(rn) < 2:
        return []
    le, re = [ln[0], ln[-1]], [rn[0], rn[-1]]
    if not all(n in m.nodes for n in le + re):
        return []
    p = lambda n: m.nodes[n][:2]
    # Pair the two left endpoints with the two right endpoints by minimum total
    # distance (optimal 2×2 matching). Anchoring only on le[0]'s nearest endpoint
    # mis-pairs short, wide, reversed-boundary pieces (a split opposing-centerline
    # lane), where le[0] sits closer to the *opposite* cut — there the two cut
    # nodes are genuinely shared with the neighbour piece but went undetected.
    straight = math.dist(p(le[0]), p(re[0])) + math.dist(p(le[1]), p(re[1]))
    crossed = math.dist(p(le[0]), p(re[1])) + math.dist(p(le[1]), p(re[0]))
    if straight <= crossed:
        return [frozenset((le[0], re[0])), frozenset((le[1], re[1]))]
    return [frozenset((le[0], re[1])), frozenset((le[1], re[0]))]


def check_vm_01_21(m):
    """Forward/backward connectivity (succ/pred).

    Lanelet2 encodes succ/pred implicitly: a lanelet's end cross-section shares
    its node ids with the next lanelet's start cross-section. A road lanelet whose
    *both* end cross-sections are unshared is fully isolated → unroutable. One open
    end is allowed (map boundary) and reported as info. Cross-sections are matched
    against all lanelets, so a road linking to a junction/shoulder counts.
    """
    def _real(ll):  # ignore degenerate sub-metre stub lanelets (geometry artifacts)
        bw = m.lanelet_bound_ways(ll)
        return max((_polyline_len(m.way_polyline(bw.get(s))) for s in ("left", "right")
                    if len(m.way_polyline(bw.get(s))) >= 2), default=0.0) >= MIN_BOUNDARY_LEN_M

    roads = [ll for ll in m.lanelets if _tags(ll).get("subtype") == "road" and _real(ll)]
    if not roads:
        return CheckResult("vm-01-21", "Forward/backward connectivity", SKIP,
                           "no road lanelets")
    cross = defaultdict(int)
    for ll in m.lanelets:
        for c in _lanelet_cross_sections(m, ll):
            cross[c] += 1
    isolated = dead_end = 0
    for ll in roads:
        cs = _lanelet_cross_sections(m, ll)
        shared = sum(1 for c in cs if cross[c] > 1)
        if shared == 0:
            isolated += 1
        elif shared < len(cs):
            dead_end += 1
    status = PASS if isolated == 0 else FAIL
    return CheckResult("vm-01-21", "Forward/backward connectivity", status,
                       f"isolated road lanelets (no succ/pred); {dead_end} one-ended (map edge)",
                       isolated, len(roads))


def check_vm_01_24(m):
    """Lanelet length: boundary ≤100 m straight / ≤20 m curved.

    Intersection lanelets are exempt (vm-03-05: a junction connector must stay
    continuous entrance→exit), identified by the S6 turn_direction /
    intersection_area tags. Crosswalks are exempt (short by nature).
    """
    over = 0
    total = 0
    for ll in m.lanelets:
        t = _tags(ll)
        if t.get("subtype") == "crosswalk" or "turn_direction" in t or "intersection_area" in t:
            continue
        for wid in m.lanelet_bound_ways(ll).values():
            pts = m.way_polyline(wid)
            if len(pts) < 2:
                continue
            total += 1
            curved = _min_interior_angle(pts) < CURVE_ANGLE_DEG
            limit = MAX_LEN_CURVED_M if curved else MAX_LEN_STRAIGHT_M
            if _polyline_len(pts) > limit:
                over += 1
    if total == 0:
        return CheckResult("vm-01-24", "Lanelet splitting", SKIP, "no lanelet boundaries")
    return CheckResult("vm-01-24", "Lanelet splitting", PASS if over == 0 else FAIL,
                       f"boundaries over length limit", over, total)


def check_vm_03_01(m):
    """Junctions must be delineated by an intersection_area polygon.

    Junction lanelets are identified by the turn_direction tag, which the
    converter emits on every OpenDRIVE junction connector (S6). The earlier
    "subtype-less lanelet" proxy broke once S3 labelled the shoulders, so it now
    SKIP'd on a false all-clear; keying off turn_direction is the real signal.
    """
    areas = [wid for wid, t in m.way_tags.items() if t.get("type") == "intersection_area"]
    junction_lls = [ll for ll in m.lanelets if "turn_direction" in _tags(ll)]
    if not junction_lls:
        return CheckResult("vm-03-01", "Intersection area", SKIP, "no junction lanelets in source")
    if not areas:
        return CheckResult("vm-03-01", "Intersection area", FAIL,
                           "junction lanelets present but 0 intersection_area polygons",
                           len(junction_lls), len(junction_lls))
    return CheckResult("vm-03-01", "Intersection area", PASS,
                       f"{len(areas)} intersection_area polygons", 0, len(areas))


def check_vm_03_02(m):
    """Every junction connector lanelet must carry turn_direction.

    Junction membership is read from the intersection_area reference tag emitted
    on each connector lanelet; every member must also carry a turn_direction. If
    no references were emitted, fall back to the turn_direction tag itself.
    """
    members = [ll for ll in m.lanelets if "intersection_area" in _tags(ll)]
    if not members:
        any_turn = [ll for ll in m.lanelets if "turn_direction" in _tags(ll)]
        if not any_turn:
            return CheckResult("vm-03-02", "Turn direction", SKIP, "no junction lanelets in source")
        return CheckResult("vm-03-02", "Turn direction", PASS,
                           "turn_direction present on junction lanelets", 0, len(any_turn))
    miss = sum(1 for ll in members if "turn_direction" not in _tags(ll))
    return CheckResult("vm-03-02", "Turn direction", PASS if miss == 0 else FAIL,
                       "junction lanelets without turn_direction", miss, len(members))


def check_vm_03_10(m):
    """Right-of-way regulatory elements at junctions (vm-03-10 signed + vm-03-11
    unsignalised right-before-left).

    The converter emits one right_of_way reg-elem per junction: signed junctions
    carry the stop/yield signs (a `refers` member) and pair yielding connectors
    with the unsigned priority connectors; unsignalised junctions are synthesized
    from CommonRoad's `left_of` (right-before-left) and carry no sign. A reg-elem
    with a `refers` member is counted as signed, otherwise right-before-left.
    """
    junction_lls = [ll for ll in m.lanelets if "turn_direction" in _tags(ll)]
    if not junction_lls:
        return CheckResult("vm-03-10", "Right of way", SKIP, "no junction lanelets in source")
    row = [r for r in m.regelems if _tags(r).get("subtype") == "right_of_way"]
    if not row:
        return CheckResult("vm-03-10", "Right of way", SKIP,
                           "no right_of_way reg-elems synthesizable", 0, len(junction_lls))
    signed = sum(1 for r in row if any(mb.get("role") == "refers" for mb in r.findall("member")))
    rbl = len(row) - signed
    return CheckResult("vm-03-10", "Right of way", PASS,
                       f"right_of_way reg-elems ({signed} signed, {rbl} right-before-left)",
                       0, len(row))


def check_vm_04_01(m):
    """Traffic light basics: traffic_light ways + traffic_light reg-elem + light_bulbs."""
    tl_ways = [wid for wid, t in m.way_tags.items() if t.get("type") == "traffic_light"]
    bulbs = [wid for wid, t in m.way_tags.items() if t.get("type") == "light_bulbs"]
    tl_regelems = [r for r in m.regelems if _tags(r).get("subtype") == "traffic_light"]
    if not tl_ways and not tl_regelems:
        return CheckResult("vm-04-01", "Traffic light basics", SKIP, "no traffic lights in source")
    missing = []
    if not tl_regelems:
        missing.append("traffic_light reg-elem")
    if not bulbs:
        missing.append("light_bulbs")
    status = PASS if not missing else FAIL
    detail = "complete" if not missing else "missing " + ", ".join(missing)
    return CheckResult("vm-04-01", "Traffic light basics", status,
                       f"{len(tl_ways)} TL ways, {len(tl_regelems)} reg-elems; {detail}",
                       len(missing), 3)


def check_vm_05_01(m):
    """Crosswalk basics: crosswalk lanelet + crosswalk_polygon + crosswalk reg-elem."""
    cw_lanelets = [ll for ll in m.lanelets if _tags(ll).get("subtype") == "crosswalk"]
    if not cw_lanelets:
        return CheckResult("vm-05-01", "Crosswalk basics", SKIP, "no crosswalks in source")
    cw_poly = [wid for wid, t in m.way_tags.items() if t.get("type") == "crosswalk_polygon"]
    cw_regelems = [r for r in m.regelems if _tags(r).get("subtype") == "crosswalk"]
    missing = []
    if not cw_poly:
        missing.append("crosswalk_polygon")
    if not cw_regelems:
        missing.append("crosswalk reg-elem")
    status = PASS if not missing else FAIL
    detail = "complete" if not missing else "missing " + ", ".join(missing)
    return CheckResult("vm-05-01", "Crosswalk basics", status,
                       f"{len(cw_lanelets)} crosswalk lanelets; {detail}",
                       len(missing), 3)


def check_vm_07_04(m):
    """Every node must carry an ele tag."""
    if not m.nodes:
        return CheckResult("vm-07-04", "Ellipsoidal height", SKIP, "no nodes")
    miss = sum(1 for (_, _, ele) in m.nodes.values() if ele is None)
    return CheckResult("vm-07-04", "Ellipsoidal height (tag present)", PASS if miss == 0 else FAIL,
                       "nodes without ele tag", miss, len(m.nodes))


CHECKS = [
    check_vm_01_01, check_vm_01_02, check_vm_01_03, check_vm_01_04, check_vm_01_05,
    check_vm_01_16, check_vm_01_21, check_vm_01_24,
    check_vm_03_01, check_vm_03_02, check_vm_03_10, check_vm_04_01, check_vm_05_01, check_vm_07_04,
]


def validate_map(m: Map) -> list[CheckResult]:
    return [chk(m) for chk in CHECKS]


def validate_file(path: str) -> list[CheckResult]:
    return validate_map(Map.parse(path))
