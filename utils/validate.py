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
CURVE_WINDOW_M = MAX_LEN_CURVED_M   # arc-length window over which curvature is measured
CURVE_TURN_DEG = 8.0           # if the heading turns more than this over any
                               # CURVE_WINDOW_M-long window the polyline is "curved"
                               # (≈143 m radius: turn°/window ≈ 1146/R, so 8°/20 m ⇒
                               # R≈143 m). Chosen so a 100 m-radius bend (~11.5°/20 m)
                               # counts as curved while a 200 m-radius one (~6°) stays
                               # straight — the midpoint of those two separates them
                               # robustly. Measured as *accumulated* turning, not a
                               # single-vertex angle, so a smoothly-sampled bend that
                               # turns tens of degrees is detected (vm-01-24).
SPLIT_LIMIT_RATIO = 0.9        # the splitter aims for pieces at this fraction of the
                               # graded limit (≈18 m curved / 90 m straight). A lanelet's
                               # left and right are cut at the *same* fractions to keep
                               # the pieces square (so their shared cut cross-sections
                               # register as connected — vm-01-21); the two boundaries
                               # differ slightly in arc length, so the longer side's
                               # pieces run a few % over the fraction target. The slack
                               # keeps them under the checker's limit.
SMOOTH_MIN_ANGLE_DEG = 60.0    # interior angle below this ⇒ a "jagged" kink (vm-01-05)
CONNECTOR_WIDTH_RATIO_MAX = 2.5  # a junction connector may flare at its mouth, but a
                                 # max/min width spread beyond this signals a geometry
                                 # error rather than a normal turn-lane taper (vm-03-03)
OVERLAP_TOL_M = 0.5              # boundaries within this trace the same physical edge
                                 # ⇒ two lanelets occupy the same strip (vm-07-08)
OVERLAP_ELE_TOL_M = 0.5         # mean-elevation gap above this ⇒ a legal multi-level
                                 # stack (overpass/tunnel), not an illegal overlap
OVERLAP_GRID_M = 5.0            # centroid-bucket cell for the O(n) candidate search


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


CURVE_RESAMPLE_STEP_M = 1.0    # resample step for the curvature metric


def _resample_polyline(pts, step):
    """Points at ~``step`` arc-length spacing along ``pts`` (endpoints preserved)."""
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + math.dist(pts[i], pts[i + 1]))
    total = cum[-1]
    if total <= 0:
        return list(pts)
    n = max(1, int(round(total / step)))
    out, j = [], 0
    for s_i in range(n + 1):
        s = total * s_i / n
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = (s - cum[j]) / seg if seg > 0 else 0.0
        out.append((pts[j][0] + t * (pts[j + 1][0] - pts[j][0]),
                    pts[j][1] + t * (pts[j + 1][1] - pts[j][1])))
    return out


def _curve_turn_over_window(pts, window):
    """Maximum accumulated heading change (deg) over any arc-length window ≤ ``window``.

    Unlike ``_min_interior_angle`` (the sharpest *single* vertex), this sums the
    turning of a whole stretch, so a gradual, finely-sampled curve — each vertex
    bending only a few degrees but turning through tens of degrees overall — is
    detected. That is the case that made vm-01-24 mis-size curved lanelets.

    The polyline is first resampled to a fixed step so the result is independent
    of the original vertex spacing. Without this, summing *discrete* per-vertex
    deflections inside an arc-length window makes the value jitter by one
    deflection increment as a cut inserts a node — a way and its own pieces could
    then disagree on curved/straight right at the threshold (vm-01-24 flapping).
    """
    if len(pts) < 3:
        return 0.0
    rs = _resample_polyline(pts, CURVE_RESAMPLE_STEP_M)
    if len(rs) < 3:
        return 0.0
    seglen, heading = [], []
    for i in range(len(rs) - 1):
        dx, dy = rs[i + 1][0] - rs[i][0], rs[i + 1][1] - rs[i][1]
        d = math.hypot(dx, dy)
        seglen.append(d)
        heading.append(math.atan2(dy, dx) if d > 0 else (heading[-1] if heading else 0.0))
    # defl[k] = turn (deg) at the interior vertex joining segment k-1 and segment k
    defl = [0.0] * len(seglen)
    for k in range(1, len(seglen)):
        a = heading[k] - heading[k - 1]
        a = (a + math.pi) % (2 * math.pi) - math.pi   # wrap to (-π, π]
        defl[k] = abs(math.degrees(a))
    # two-pointer window over segments [lo..hi]; run = Σ defl at interior vertices (lo, hi]
    best = run = length = 0.0
    lo = 0
    for hi in range(len(seglen)):
        length += seglen[hi]
        run += defl[hi]
        while length > window and lo < hi:
            length -= seglen[lo]
            run -= defl[lo + 1]
            lo += 1
        best = max(best, run)
    return best


def _length_limit(pts):
    """Max allowed boundary length (m): ``MAX_LEN_CURVED_M`` if the polyline curves
    (heading turns more than ``CURVE_TURN_DEG`` over any ``CURVE_WINDOW_M`` window),
    else ``MAX_LEN_STRAIGHT_M``. The single classifier shared by the splitter
    (``utils.split``) and the checker (``check_vm_01_24``) so they cannot diverge.

    Whole-polyline verdict: used by the *checker* on each already-split piece
    (which is single-class by construction — see ``_adaptive_cut_fractions``) and
    as a coarse "does this way need splitting at all" gate.
    """
    curved = _curve_turn_over_window(pts, CURVE_WINDOW_M) > CURVE_TURN_DEG
    return MAX_LEN_CURVED_M if curved else MAX_LEN_STRAIGHT_M


def _curve_mask(rs, thr=CURVE_TURN_DEG):
    """Per-segment curved/straight mask for a resampled polyline ``rs``.

    Segment ``i`` (between ``rs[i]`` and ``rs[i+1]``) is *curved* if it lies in
    some arc-length window ≤ ``CURVE_WINDOW_M`` whose accumulated heading change
    exceeds ``thr`` — the same "any window turns more than the threshold" test as
    ``_curve_turn_over_window``, but resolved locally per segment instead of
    collapsed to a single verdict for the whole way. This is what lets a straight
    tail and a curved middle on one boundary get different split spacings.
    """
    n = len(rs) - 1
    if n < 1:
        return []
    seg, heading = [], []
    for i in range(n):
        dx, dy = rs[i + 1][0] - rs[i][0], rs[i + 1][1] - rs[i][1]
        d = math.hypot(dx, dy)
        seg.append(d)
        heading.append(math.atan2(dy, dx) if d > 0 else (heading[-1] if heading else 0.0))
    # defl[k] = turn (deg) at the interior vertex joining segment k-1 and k
    defl = [0.0] * n
    for k in range(1, n):
        a = heading[k] - heading[k - 1]
        a = (a + math.pi) % (2 * math.pi) - math.pi
        defl[k] = abs(math.degrees(a))
    curved = [False] * n
    for a in range(n):
        length = run = 0.0
        for b in range(a, n):
            length += seg[b]
            if b > a:
                run += defl[b]
            if length > CURVE_WINDOW_M:
                break
            if run > thr:
                for t in range(a, b + 1):
                    curved[t] = True
                break
    return curved


def _dilate_mask(mask, seg, radius):
    """Grow the True regions of a per-segment boolean ``mask`` outward by
    ``radius`` arc-length (a 1-D distance transform along the chain). Used to widen
    the curved zone so cut spacing is already tight on the *approach* to a curve —
    otherwise a piece straddling the straight→curve transition can be both long and
    curve-classified, which would fail the per-piece length check."""
    n = len(mask)
    if n == 0 or not any(mask):
        return list(mask)
    INF = float("inf")
    d = [0.0 if m else INF for m in mask]
    for i in range(1, n):                       # forward sweep
        if d[i] > 0:
            step = (seg[i - 1] + seg[i]) / 2.0
            d[i] = min(d[i], d[i - 1] + step)
    for i in range(n - 2, -1, -1):              # backward sweep
        if d[i] > 0:
            step = (seg[i] + seg[i + 1]) / 2.0
            d[i] = min(d[i], d[i + 1] + step)
    return [dist <= radius for dist in d]


def _local_limits(pts):
    """Resample ``pts`` and return ``(seg_lengths, per_seg_limit)`` where each
    segment's limit is ``MAX_LEN_CURVED_M`` inside a curved stretch (see
    ``_curve_mask``, widened by one window) and ``MAX_LEN_STRAIGHT_M`` on a
    straight one."""
    rs = _resample_polyline(pts, CURVE_RESAMPLE_STEP_M)
    seg = [math.dist(rs[i], rs[i + 1]) for i in range(len(rs) - 1)]
    curved = _dilate_mask(_curve_mask(rs, CURVE_TURN_DEG), seg, CURVE_WINDOW_M)
    lim = [(MAX_LEN_CURVED_M if c else MAX_LEN_STRAIGHT_M) * SPLIT_LIMIT_RATIO
           for c in curved]
    return seg, lim


def _combined_cut_fractions(polys, samples=400):
    """Interior arc-length fractions to cut a whole *group* of boundaries at, in a
    common [0, 1] parameter (fraction of each way's own length).

    The group is a connected lane component — a lanelet's two boundaries plus every
    way they transitively share. All are cut at the SAME fractions so the pieces
    stay square and their shared cut cross-sections register as connected. The cut
    density at parameter ``f`` is ``max`` over the group's ways of
    ``length / local_limit(f)``: wherever *any* way in the group is long or curved,
    the cuts there are tight enough for it, so every way's pieces land within their
    own straight/curved limit — including a taper/merge where a bend sits at a
    different fraction on each side. Equal *cumulative* density between cuts then
    spaces them densely through curves and sparsely along straights.

    Returns ``[]`` when the group needs no cut (combined weight ≤ 1)."""
    profs = []
    for pts in polys:
        if len(pts) < 2:
            continue
        seg, lim = _local_limits(pts)
        total = sum(seg)
        if total <= 0:
            continue
        cum = [0.0]
        for s in seg:
            cum.append(cum[-1] + s)
        samp, j = [], 0
        for i in range(samples):
            pos = ((i + 0.5) / samples) * total
            while j < len(seg) - 1 and cum[j + 1] <= pos:
                j += 1
            samp.append(total / lim[j])          # pieces-per-unit-f this way demands
        profs.append(samp)
    if not profs:
        return []
    df = 1.0 / samples
    dens = [max(p[i] for p in profs) for i in range(samples)]
    W = sum(dens) * df
    if W <= 1 + 1e-9:
        return []
    k = math.ceil(W - 1e-9)
    fracs, ti, cw = [], 1, 0.0
    for i in range(samples):
        cw0 = cw
        cw += dens[i] * df
        while ti < k and cw >= W * ti / k - 1e-12:
            wt = W * ti / k
            t = (wt - cw0) / (cw - cw0) if cw > cw0 else 0.0
            fracs.append((i + t) / samples)
            ti += 1
    return [f for f in fracs if 1e-6 < f < 1 - 1e-6]


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


BORDER_COINCIDENT_TOL_M = 0.50    # boundary ways within this trace the same physical line
                                  # (matches vm-01-04's coincidence tol)


def _coincident_boundary_ids(m, tol=BORDER_COINCIDENT_TOL_M):
    """Boundary ways that trace the same physical line as another boundary way.

    Mirrors utils/road_border.py: two opposing road lanelets that each keep
    their own (unmerged) way for the shared centerline both look like outer
    edges by reference count, but they are a crossable interior divider. A true
    outer edge has no coincident boundary twin. Endpoint-bucketed (1 m) so the
    pairwise test stays ~O(n); a reversed polyline lands in the same bucket.
    """
    polys = {}
    for ll in m.lanelets:
        for wid in m.lanelet_bound_ways(ll).values():
            if wid in polys:
                continue
            pts = m.way_polyline(wid)
            if len(pts) >= 2 and _polyline_len(pts) >= MIN_BOUNDARY_LEN_M:
                polys[wid] = pts
    buckets = defaultdict(list)
    for wid, pts in polys.items():
        p0 = (round(pts[0][0]), round(pts[0][1]))
        p1 = (round(pts[-1][0]), round(pts[-1][1]))
        buckets[tuple(sorted((p0, p1)))].append(wid)
    hit = set()
    for group in buckets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _polylines_equal(polys[group[i]], polys[group[j]], tol):
                    hit.add(group[i])
                    hit.add(group[j])
    return hit


def check_vm_01_02(m):
    """Every boundary line must carry a `type` and `lane_change`.

    The requirement: lines (`way`) must have `type` (`line_thin`/`line_thick`/
    `road_border`) and `lane_change` (painted lines also carry `subtype`). Only
    drivable-lanelet boundaries (`subtype:road`/`road_shoulder`) are graded — a
    sidewalk/crosswalk edge is not a lane line. Crossable unmarked dividers
    between two `subtype:road` lanelets are exempt (the source authors no
    marking and the divider is genuinely crossable; left untyped by policy, see
    utils/road_border.py) — whether modelled as one shared way or as two
    coincident-but-unmerged ways.
    """
    refs = defaultdict(list)               # boundary way id -> neighbour subtypes
    for ll in m.lanelets:
        sub = _tags(ll).get("subtype")
        for wid in m.lanelet_bound_ways(ll).values():
            refs[wid].append(sub)
    if not refs:
        return CheckResult("vm-01-02", "Boundary line tags", SKIP, "no boundary ways")
    coincident = _coincident_boundary_ids(m)
    miss = total = 0
    for wid, subs in refs.items():
        if not any(s in ("road", "road_shoulder") for s in subs):
            continue                       # non-carriageway boundary (sidewalk/crosswalk) — not graded
        if len(subs) >= 2 and all(s == "road" for s in subs):
            continue                       # crossable unmarked interior divider — exempt
        if wid in coincident:
            continue                       # coincident-but-unmerged opposing centerline — exempt
        total += 1
        t = m.way_tags.get(wid, {})
        if t.get("type") not in ("line_thin", "line_thick", "road_border") \
                or "lane_change" not in t:
            miss += 1
    return CheckResult("vm-01-02", "Boundary line tags", PASS if miss == 0 else FAIL,
                       "boundary lines missing type/lane_change", miss, total)


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


def _splittable_lanelets(m):
    """Replicate split.py's cuttability fixpoint.

    A lanelet is exempt (never split) if it lacks a boundary or is a walkway /
    crosswalk / intersection connector. Then, monotonically: a way is *cuttable*
    only if every lanelet owning it is splittable, and a lanelet is splittable
    only if both its ways are cuttable. This propagates un-cuttability through
    shared boundaries (a road_shoulder sharing one way with an exempt walkway
    cannot be split, which in turn pins the road it shares its other boundary
    with). Returns (splittable: {ll_id: bool}, bounds: {ll_id: (left, right)}).
    """
    bounds = {}
    exempt = {}
    owners = defaultdict(list)
    for ll in m.lanelets:
        bw = m.lanelet_bound_ways(ll)
        left, right = bw.get("left"), bw.get("right")
        lid = id(ll)
        bounds[lid] = (left, right)
        t = _tags(ll)
        exempt[lid] = (left is None or right is None
                       or t.get("subtype") in ("crosswalk", "walkway")
                       or "turn_direction" in t or "intersection_area" in t)
        for wid in (left, right):
            if wid is not None:
                owners[wid].append(lid)
    splittable = {lid: not exempt[lid] for lid in bounds}
    changed = True
    while changed:
        changed = False
        cuttable = {wid: all(splittable[o] for o in own) for wid, own in owners.items()}
        for lid, (left, right) in bounds.items():
            if exempt[lid]:
                continue
            ok = cuttable.get(left, False) and cuttable.get(right, False)
            if splittable[lid] != ok:
                splittable[lid] = ok
                changed = True
    return splittable, exempt, owners


def check_vm_01_24(m):
    """Lanelet length: boundary ≤100 m straight / ≤20 m curved.

    Intersection lanelets are exempt (vm-03-05: a junction connector must stay
    continuous entrance→exit), identified by the S6 turn_direction /
    intersection_area tags. Crosswalks and walkways are exempt: Autoware imposes
    no length limit on pedestrian lanelets and forbids splitting them (its
    longitudinal_subtype_connection validator bans a walkway/crosswalk successor),
    so split.py keeps them whole.

    Only genuinely *cuttable* boundaries are graded: an over-length way that
    split.py cannot cut without also splitting an exempt lanelet (a boundary
    shared, directly or transitively, with a walkway/crosswalk/intersection) is
    un-cuttable by design and reported separately, not as a failure.
    """
    splittable, exempt, owners = _splittable_lanelets(m)
    cuttable = {wid: all(splittable[o] for o in own) for wid, own in owners.items()}
    over = uncuttable = total = 0
    for wid, own in owners.items():
        if all(exempt[o] for o in own):
            continue                       # pedestrian/junction boundary — not length-graded
        pts = m.way_polyline(wid)
        if len(pts) < 2:
            continue
        total += 1
        if _polyline_len(pts) > _length_limit(pts):
            if cuttable.get(wid, False):
                over += 1
            else:
                uncuttable += 1
    if total == 0:
        return CheckResult("vm-01-24", "Lanelet splitting", SKIP, "no lanelet boundaries")
    detail = "cuttable boundaries over length limit"
    if uncuttable:
        detail += f" ({uncuttable} more un-cuttable, shared with exempt lanelets)"
    return CheckResult("vm-01-24", "Lanelet splitting", PASS if over == 0 else FAIL,
                       detail, over, total)


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


def _connector_lanelets(m):
    """Junction connector lanelets — those the S6 pass tagged with turn_direction."""
    return [ll for ll in m.lanelets if "turn_direction" in _tags(ll)]


def check_vm_07_06(m):
    """Junction lanelet completeness (vm-03-04 / vm-07-06).

    'Create all intersection lanelets (incl. additional lanes)': the failure mode
    is a dropped junction connector, which surfaces as a connector with a dangling
    end (its missing neighbour was the lane that should have been created). Using
    the same shared-cross-section signal as vm-01-21, a connector is complete iff
    BOTH its end cross-sections are shared with another lanelet (entry present and
    exit present); an open end means an adjoining lanelet is absent.
    """
    connectors = _connector_lanelets(m)
    if not connectors:
        return CheckResult("vm-07-06", "Junction lanelet completeness", SKIP,
                           "no junction lanelets in source")
    xsec_owners = defaultdict(int)
    for ll in m.lanelets:
        for cs in _lanelet_cross_sections(m, ll):
            xsec_owners[cs] += 1
    incomplete = 0
    for ll in connectors:
        cs = _lanelet_cross_sections(m, ll)
        # a connector needs two resolvable cross-sections, each shared with
        # at least one other lanelet (the incoming approach and the outgoing road)
        if len(cs) < 2 or any(xsec_owners[c] < 2 for c in cs):
            incomplete += 1
    return CheckResult("vm-07-06", "Junction lanelet completeness",
                       PASS if incomplete == 0 else FAIL,
                       "connectors missing an entry/exit neighbour",
                       incomplete, len(connectors))


def _point_in_ring(pt, ring):
    """Ray-cast point-in-polygon test; ``ring`` is a list of (x, y) (auto-closed)."""
    x, y = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def check_vm_03_08(m):
    """Intersection area range (vm-03-08): the polygon must cover its connectors.

    vm-03-01 only checks an intersection_area exists; this checks it actually
    bounds the junction's range — every connector that references an
    intersection_area must have its mid-station centerline point inside that
    polygon. That point (mean of the two boundary midpoints) sits squarely on the
    lanelet, so it is stable under the downsample/split that runs after emission;
    the plain boundary centroid is not — on a tight turn it lands inside the curve,
    off the lanelet. A connector outside its area means the range is mis-delineated.
    """
    areas = {wid: m.way_polyline(wid) for wid, t in m.way_tags.items()
             if t.get("type") == "intersection_area"}
    connectors = _connector_lanelets(m)
    if not connectors:
        return CheckResult("vm-03-08", "Intersection area range", SKIP,
                           "no junction lanelets in source")
    if not areas:
        return CheckResult("vm-03-08", "Intersection area range", SKIP,
                           "no intersection_area polygons (presence is vm-03-01)")
    offenders = total = 0
    for ll in connectors:
        ring = areas.get(_tags(ll).get("intersection_area"))
        if not ring or len(ring) < 4:
            continue
        bw = m.lanelet_bound_ways(ll)
        lp, rp = m.way_polyline(bw.get("left")), m.way_polyline(bw.get("right"))
        if len(lp) < 2 or len(rp) < 2:
            continue
        lm, rm = _point_at_fraction(lp, 0.5), _point_at_fraction(rp, 0.5)
        midpoint = ((lm[0] + rm[0]) / 2, (lm[1] + rm[1]) / 2)
        total += 1
        if not _point_in_ring(midpoint, ring):
            offenders += 1
    return CheckResult("vm-03-08", "Intersection area range",
                       PASS if offenders == 0 else FAIL,
                       "connectors outside their intersection_area", offenders, total)


STOP_LINE_RANGE_TOL_M = 6.0   # a stop line bounding a junction approach sits within
                              # this of the governed lanelet's junction-facing end
                              # (a small setback from the entry cross-section)


def check_vm_03_09(m):
    """Intersection lanelet range bounded by stop line (vm-03-09).

    A junction approach that must stop carries a right_of_way reg-elem binding the
    yielding lanelet to a stop_line (``ref_line``). The stop line marks where the
    junction range begins, so it must sit at the bound lanelet's junction-facing
    cross-section. This flags reg-elem-governed stop lines positioned more than
    STOP_LINE_RANGE_TOL_M from any cross-section of a lanelet their reg-elem
    references. Source-dependent: approaches with no source stop line cannot be
    bounded and are not counted (SKIP when there are none).
    """
    stop_lines = {wid: m.way_polyline(wid) for wid, t in m.way_tags.items()
                  if t.get("type") == "stop_line"}
    if not stop_lines:
        return CheckResult("vm-03-09", "Stop-line-bounded range", SKIP,
                           "no stop lines in source")
    ll_ends = {}
    for ll in m.lanelets:
        bw = m.lanelet_bound_ways(ll)
        lp, rp = m.way_polyline(bw.get("left")), m.way_polyline(bw.get("right"))
        if len(lp) < 2 or len(rp) < 2:
            continue
        ll_ends[ll.get("id")] = (
            ((lp[0][0] + rp[0][0]) / 2, (lp[0][1] + rp[0][1]) / 2),
            ((lp[-1][0] + rp[-1][0]) / 2, (lp[-1][1] + rp[-1][1]) / 2),
        )
    mis = total = 0
    for r in m.regelems:
        ref_sl = [mb.get("ref") for mb in r.findall("member")
                  if mb.get("role") == "ref_line" and mb.get("ref") in stop_lines]
        if not ref_sl:
            continue
        cross = [e for mb in r.findall("member") if mb.get("type") == "relation"
                 for e in ll_ends.get(mb.get("ref"), ())]
        if not cross:
            continue
        for s in ref_sl:
            smp = _point_at_fraction(stop_lines[s], 0.5)
            total += 1
            if min(math.dist(smp, e) for e in cross) > STOP_LINE_RANGE_TOL_M:
                mis += 1
    if total == 0:
        return CheckResult("vm-03-09", "Stop-line-bounded range", SKIP,
                           "no reg-elem-governed stop lines")
    return CheckResult("vm-03-09", "Stop-line-bounded range",
                       PASS if mis == 0 else FAIL,
                       "governed stop lines not at the junction range boundary",
                       mis, total)


def _point_at_fraction(pts, f):
    """Point at arc-length fraction ``f`` in [0, 1] along a polyline."""
    if len(pts) == 1:
        return pts[0]
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    if total == 0:
        return pts[0]
    target = f * total
    acc = 0.0
    for i, s in enumerate(seg):
        if acc + s >= target:
            t = (target - acc) / s if s else 0.0
            return (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
        acc += s
    return pts[-1]


def _connector_widths(m, ll, n=7):
    """Lanelet width sampled at ``n`` equal arc-length stations (left↔right gap)."""
    bw = m.lanelet_bound_ways(ll)
    lp, rp = m.way_polyline(bw.get("left")), m.way_polyline(bw.get("right"))
    if len(lp) < 2 or len(rp) < 2:
        return []
    # align orientation so station k pairs the same end of both boundaries
    if (math.dist(lp[0], rp[0]) + math.dist(lp[-1], rp[-1])
            > math.dist(lp[0], rp[-1]) + math.dist(lp[-1], rp[0])):
        rp = rp[::-1]
    return [math.dist(_point_at_fraction(lp, k / (n - 1)), _point_at_fraction(rp, k / (n - 1)))
            for k in range(n)]


CONNECTOR_SHARE_TOL_M = 0.1   # two connector boundaries within this over their full
                              # length are the same physical line ⇒ should be one way


def _polyline_gap(a, b, n=5):
    """Orientation-independent max distance between two polylines sampled at ``n``
    arc-length stations — a proxy for 'are these the same physical line'."""
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    fa = [_point_at_fraction(a, k / (n - 1)) for k in range(n)]
    fb = [_point_at_fraction(b, k / (n - 1)) for k in range(n)]
    fwd = max(math.dist(fa[k], fb[k]) for k in range(n))
    rev = max(math.dist(fa[k], fb[n - 1 - k]) for k in range(n))
    return min(fwd, rev)


def check_vm_03_07(m):
    """Adjacent connecting lanes must share their dividing linestring (vm-03-07).

    odr2cr leaves connector adjacency unset, so S4's adjacency-based sharing never
    fires for junction connectors and only its ~1 mm equal-vertex-count geometric
    fallback does. This flags connector-boundary pairs within one junction that
    coincide over their whole length (within CONNECTOR_SHARE_TOL_M) yet were
    emitted as two distinct ways — an unmerged shared line. Genuinely gapped
    adjacent lanes (a real lane-marking gap) are correctly not merged and not
    counted; the residual here is the same coincident-but-mismatched-vertex-count
    class accepted for vm-01-04 / vm-01-16.
    """
    connectors = _connector_lanelets(m)
    if not connectors:
        return CheckResult("vm-03-07", "Adjacent connector sharing", SKIP,
                           "no junction lanelets in source")
    by_area = defaultdict(set)
    for ll in connectors:
        area = _tags(ll).get("intersection_area")
        for wid in m.lanelet_bound_ways(ll).values():
            by_area[area].add(wid)
    unmerged = 0
    total = 0
    for ways in by_area.values():
        wl = [w for w in ways if _polyline_len(m.way_polyline(w)) > MIN_BOUNDARY_LEN_M]
        total += len(wl)
        polys = {w: m.way_polyline(w) for w in wl}
        for i in range(len(wl)):
            for j in range(i + 1, len(wl)):
                if _polyline_gap(polys[wl[i]], polys[wl[j]]) < CONNECTOR_SHARE_TOL_M:
                    unmerged += 1
    return CheckResult("vm-03-07", "Adjacent connector sharing",
                       PASS if unmerged == 0 else FAIL,
                       "coincident connector boundaries emitted as separate ways",
                       unmerged, total)


def check_vm_03_03(m):
    """Intersection width/shape (vm-03-03): connector width consistent + curves smooth.

    Two geometry defects are counted over junction connectors: an implausible
    width spread (max/min sampled width above CONNECTOR_WIDTH_RATIO_MAX — beyond a
    normal turn-lane taper), and a jagged boundary (interior kink below
    SMOOTH_MIN_ANGLE_DEG, the vm-01-05 smoothness threshold). The worst of the two
    counts gates the result.
    """
    connectors = _connector_lanelets(m)
    if not connectors:
        return CheckResult("vm-03-03", "Intersection width/shape", SKIP,
                           "no junction lanelets in source")
    bad_width = bad_smooth = 0
    for ll in connectors:
        w = _connector_widths(m, ll)
        if w and min(w) > 0 and max(w) / min(w) > CONNECTOR_WIDTH_RATIO_MAX:
            bad_width += 1
        bw = m.lanelet_bound_ways(ll)
        for s in ("left", "right"):
            pts = m.way_polyline(bw.get(s))
            if len(pts) >= 3 and _min_interior_angle(pts) < SMOOTH_MIN_ANGLE_DEG:
                bad_smooth += 1
                break
    worst = max(bad_width, bad_smooth)
    return CheckResult("vm-03-03", "Intersection width/shape",
                       PASS if worst == 0 else FAIL,
                       f"width-jump={bad_width} jagged={bad_smooth}",
                       worst, len(connectors))


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


def _sample_n(pts, n):
    """``n`` points (n≥2) evenly spaced by arc length along ``pts``, endpoints kept.

    Unlike ``_resample_polyline`` (fixed *step* ⇒ count depends on length) this
    yields a fixed *count*, so two polylines of the same edge but different vertex
    spacing become directly comparable point-for-point.
    """
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + math.dist(pts[i], pts[i + 1]))
    total = cum[-1]
    if total <= 0:
        return [tuple(pts[0])] * n
    out, j = [], 0
    for i in range(n):
        s = total * i / (n - 1)
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = (s - cum[j]) / seg if seg > 0 else 0.0
        out.append((pts[j][0] + t * (pts[j + 1][0] - pts[j][0]),
                    pts[j][1] + t * (pts[j + 1][1] - pts[j][1])))
    return out


def _polylines_coincident(a, b, tol, n=12):
    """True if ``a`` and ``b`` trace the same physical edge (either direction).

    Vertex-count independent (both resampled to ``n`` points), with a length-ratio
    pre-filter so a short stub cannot match a long boundary that happens to start
    nearby.
    """
    if len(a) < 2 or len(b) < 2:
        return False
    la, lb = _polyline_len(a), _polyline_len(b)
    if min(la, lb) < MIN_BOUNDARY_LEN_M or min(la, lb) / max(la, lb) < 0.9:
        return False
    sa, sb = _sample_n(a, n), _sample_n(b, n)
    fwd = max(math.dist(sa[i], sb[i]) for i in range(n))
    rev = max(math.dist(sa[i], sb[n - 1 - i]) for i in range(n))
    return min(fwd, rev) < tol


def _way_mean_ele(m, wid):
    eles = [m.nodes[nd][2] for nd in m.way_nodes.get(wid, [])
            if nd in m.nodes and m.nodes[nd][2] is not None]
    return sum(eles) / len(eles) if eles else None


def _lanelet_dir(lp, rp):
    """Unit travel-direction vector (start→end cross-section midpoints)."""
    sx, sy = (lp[0][0] + rp[0][0]) / 2, (lp[0][1] + rp[0][1]) / 2
    ex, ey = (lp[-1][0] + rp[-1][0]) / 2, (lp[-1][1] + rp[-1][1]) / 2
    dx, dy = ex - sx, ey - sy
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 0 else None


def check_vm_07_08(m):
    """Overlapping lanes: no two lanelets occupy the *same* strip.

    A full overlap is two lanelets whose *both* boundaries coincide (same left+right
    edge, in either assignment) — a duplicated lane. Lateral neighbours share only
    one boundary and successors share only a cross-section, so neither registers.
    Two legal exceptions are excused (vm-07-08 wording): a **multi-level** stack
    (mean elevations differ by > OVERLAP_ELE_TOL_M — an overpass/tunnel over the
    same ground plan) and a **bidirectional** pair (the two lanelets traverse the
    strip in opposite directions, or either is tagged one_way=no). Anything left is
    an illegal same-level, same-direction duplicate.

    Candidate pairs are found via a centroid grid, so the scan is ~O(n).
    Scope is **non-junction drivable lanes** (subtype road). Two exclusions:
    walkways/road_shoulders/crosswalks legitimately trace the same edge as the
    adjacent furniture (a sidewalk and a shoulder stacked on one narrow strip);
    and intersection lanelets (``turn_direction`` / ``intersection_area``) are
    exempt because several maneuvers legally share one junction-mouth stub — the
    same exemption vm-01-24 / vm-03-05 make. Autoware disambiguates those via
    turn_direction + right_of_way, so overlapping connectors are not a defect.
    """
    items = []
    for ll in m.lanelets:
        t = _tags(ll)
        if t.get("subtype") != "road":
            continue
        if "turn_direction" in t or "intersection_area" in t:
            continue
        bw = m.lanelet_bound_ways(ll)
        lw, rw = bw.get("left"), bw.get("right")
        lp, rp = m.way_polyline(lw), m.way_polyline(rw)
        if len(lp) < 2 or len(rp) < 2:
            continue
        if max(_polyline_len(lp), _polyline_len(rp)) < MIN_BOUNDARY_LEN_M:
            continue
        allpts = lp + rp
        cx = sum(p[0] for p in allpts) / len(allpts)
        cy = sum(p[1] for p in allpts) / len(allpts)
        ele = [e for e in (_way_mean_ele(m, lw), _way_mean_ele(m, rw)) if e is not None]
        items.append({
            "lp": lp, "rp": rp, "cx": cx, "cy": cy,
            "ele": sum(ele) / len(ele) if ele else None,
            "dir": _lanelet_dir(lp, rp),
            "one_way": t.get("one_way"),
        })
    if len(items) < 2:
        return CheckResult("vm-07-08", "Overlapping lanes", SKIP, "fewer than two lanelets")

    buckets = defaultdict(list)
    for idx, it in enumerate(items):
        buckets[(round(it["cx"] / OVERLAP_GRID_M), round(it["cy"] / OVERLAP_GRID_M))].append(idx)

    illegal = overlaps = 0
    checked = set()
    for (gx, gy), idxs in buckets.items():
        cand = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for j in buckets.get((gx + dx, gy + dy), [])]
        for i in idxs:
            a = items[i]
            for j in cand:
                if j <= i:
                    continue
                key = (i, j)
                if key in checked:
                    continue
                checked.add(key)
                b = items[j]
                if math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) > OVERLAP_GRID_M:
                    continue
                same = (_polylines_coincident(a["lp"], b["lp"], OVERLAP_TOL_M)
                        and _polylines_coincident(a["rp"], b["rp"], OVERLAP_TOL_M))
                crossed = (_polylines_coincident(a["lp"], b["rp"], OVERLAP_TOL_M)
                           and _polylines_coincident(a["rp"], b["lp"], OVERLAP_TOL_M))
                if not (same or crossed):
                    continue
                overlaps += 1
                # legal exception 1: multi-level stack
                if a["ele"] is not None and b["ele"] is not None \
                        and abs(a["ele"] - b["ele"]) > OVERLAP_ELE_TOL_M:
                    continue
                # legal exception 2: bidirectional (opposing dir, or declared two-way)
                opposing = (a["dir"] and b["dir"]
                            and a["dir"][0] * b["dir"][0] + a["dir"][1] * b["dir"][1] < 0)
                if opposing or a["one_way"] == "no" or b["one_way"] == "no":
                    continue
                illegal += 1

    return CheckResult("vm-07-08", "Overlapping lanes", PASS if illegal == 0 else FAIL,
                       f"illegal full overlaps ({overlaps} overlapping pairs, "
                       f"multi-level/bidirectional excused)", illegal, len(items))


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
    check_vm_03_01, check_vm_03_02, check_vm_03_03, check_vm_03_07, check_vm_03_08,
    check_vm_03_09, check_vm_03_10, check_vm_04_01, check_vm_05_01,
    check_vm_07_04, check_vm_07_06, check_vm_07_08,
]


def validate_map(m: Map) -> list[CheckResult]:
    return [chk(m) for chk in CHECKS]


def validate_file(path: str) -> list[CheckResult]:
    return validate_map(Map.parse(path))
