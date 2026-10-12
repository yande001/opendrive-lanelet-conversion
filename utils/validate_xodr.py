"""Structural checks for a generated OpenDRIVE (.xodr) file.

Pure-function checks over the XML, in the PASS/FAIL/SKIP ``CheckResult`` style of
utils/validate.py (no external validator needed). They catch the failure modes the
Lanelet2 -> OpenDRIVE pipeline can actually produce: a projection string consumers cannot
parse, broken plan-view bookkeeping, dangling road/junction links, lanes without widths and
gaps between consecutive geometry primitives.

    from utils.validate_xodr import validate_file
    for r in validate_file("output/r01.xodr"): print(r.line())
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from lxml import etree

from utils.validate import PASS, FAIL, SKIP, CheckResult

LENGTH_SUM_TOL_M = 1e-3
CONTINUITY_TOL_M = 0.05
ROADMARK_WIDTH_MAX_M = 1.0


class Xodr:
    """Parsed .xodr with the lookups the checks share."""

    def __init__(self, root: etree._Element):
        self.root = root
        self.header = root.find("header")
        self.roads = root.findall("road")
        self.junctions = root.findall("junction")
        self.road_by_id = {r.get("id"): r for r in self.roads}
        self.junction_by_id = {j.get("id"): j for j in self.junctions}

    @classmethod
    def parse(cls, path) -> "Xodr":
        return cls(etree.parse(str(path)).getroot())

    @staticmethod
    def lanes_of(road: etree._Element) -> dict[str, etree._Element]:
        out = {}
        for lane in road.findall("lanes/laneSection/*/lane"):
            out[lane.get("id")] = lane
        return out


def _f(el: etree._Element, attr: str) -> float:
    try:
        return float(el.get(attr))
    except (TypeError, ValueError):
        return float("nan")


# --- geometry evaluation (for continuity) -----------------------------------------------

def _geometry_end(geom: etree._Element) -> tuple[float, float] | None:
    """End point of one <geometry> primitive, or None if the type is unsupported."""
    x, y, hdg, length = (_f(geom, a) for a in ("x", "y", "hdg", "length"))
    if any(math.isnan(v) for v in (x, y, hdg, length)):
        return None
    child = geom[0] if len(geom) else None
    if child is None:
        return None
    if child.tag == "line":
        return x + length * math.cos(hdg), y + length * math.sin(hdg)
    if child.tag == "arc":
        k = _f(child, "curvature")
        if abs(k) < 1e-12:
            return x + length * math.cos(hdg), y + length * math.sin(hdg)
        theta = k * length
        dx = (math.sin(hdg + theta) - math.sin(hdg)) / k
        dy = (-math.cos(hdg + theta) + math.cos(hdg)) / k
        return x + dx, y + dy
    if child.tag == "spiral":
        k0, k1 = _f(child, "curvStart"), _f(child, "curvEnd")
        # numerical integration of the clothoid (fine for the short pieces cr2odr emits)
        n = max(8, int(length / 0.1))
        s = np.linspace(0.0, length, n + 1)
        kappa = k0 + (k1 - k0) * s / length if length > 0 else np.full_like(s, k0)
        theta = hdg + np.concatenate([[0.0], np.cumsum((kappa[1:] + kappa[:-1]) / 2 * np.diff(s))])
        xs = x + np.concatenate([[0.0], np.cumsum((np.cos(theta[1:]) + np.cos(theta[:-1])) / 2 * np.diff(s))])
        ys = y + np.concatenate([[0.0], np.cumsum((np.sin(theta[1:]) + np.sin(theta[:-1])) / 2 * np.diff(s))])
        return float(xs[-1]), float(ys[-1])
    if child.tag == "paramPoly3":
        a_u, b_u, c_u, d_u = (_f(child, k) for k in ("aU", "bU", "cU", "dU"))
        a_v, b_v, c_v, d_v = (_f(child, k) for k in ("aV", "bV", "cV", "dV"))
        p = 1.0 if child.get("pRange", "normalized") == "normalized" else length
        u = a_u + b_u * p + c_u * p ** 2 + d_u * p ** 3
        v = a_v + b_v * p + c_v * p ** 2 + d_v * p ** 3
        return x + u * math.cos(hdg) - v * math.sin(hdg), y + u * math.sin(hdg) + v * math.cos(hdg)
    return None


# --- checks ------------------------------------------------------------------------------

def check_header(x: Xodr) -> CheckResult:
    if x.root.tag != "OpenDRIVE" or x.header is None:
        return CheckResult("xodr-01", "header", FAIL, "root is not <OpenDRIVE> with a <header>")
    major, minor = x.header.get("revMajor"), x.header.get("revMinor")
    if major != "1" or minor is None:
        return CheckResult("xodr-01", "header", FAIL, f"revMajor/revMinor = {major}/{minor}")
    return CheckResult("xodr-01", "header", PASS, f"OpenDRIVE {major}.{minor}, {len(x.roads)} roads, "
                       f"{len(x.junctions)} junctions")


def check_georeference(x: Xodr) -> CheckResult:
    geo = x.header.find("geoReference") if x.header is not None else None
    if geo is None or not (geo.text or "").strip():
        return CheckResult("xodr-02", "geoReference parses", SKIP, "no <geoReference>")
    text = geo.text.strip()
    if "CDATA" in text:
        return CheckResult("xodr-02", "geoReference parses", FAIL,
                           "literal '<![CDATA[' inside the text (escaped CDATA)")
    try:
        from pyproj import CRS
        CRS(text)
    except Exception as e:  # noqa: BLE001
        return CheckResult("xodr-02", "geoReference parses", FAIL, f"pyproj rejects it: {e}")
    return CheckResult("xodr-02", "geoReference parses", PASS, text[:70])


def check_road_ids_unique(x: Xodr) -> CheckResult:
    dup = [k for k, c in Counter(r.get("id") for r in x.roads).items() if c > 1]
    if dup:
        return CheckResult("xodr-03", "road ids unique", FAIL, f"duplicates: {dup[:5]}", len(dup), len(x.roads))
    return CheckResult("xodr-03", "road ids unique", PASS, "", 0, len(x.roads))


def check_planview(x: Xodr) -> CheckResult:
    bad = []
    for r in x.roads:
        geoms = r.findall("planView/geometry")
        length = _f(r, "length")
        if not geoms or math.isnan(length) or length <= 0:
            bad.append(r.get("id"))
            continue
        s_vals = [_f(g, "s") for g in geoms]
        lens = [_f(g, "length") for g in geoms]
        if any(math.isnan(v) for v in s_vals + lens) or any(v <= 0 for v in lens):
            bad.append(r.get("id"))
            continue
        if any(s_vals[i + 1] < s_vals[i] for i in range(len(s_vals) - 1)):
            bad.append(r.get("id"))
            continue
        if abs(sum(lens) - length) > LENGTH_SUM_TOL_M or abs(s_vals[-1] + lens[-1] - length) > LENGTH_SUM_TOL_M:
            bad.append(r.get("id"))
            continue
        if any(math.isnan(_f(g, a)) for g in geoms for a in ("x", "y", "hdg")):
            bad.append(r.get("id"))
    status = FAIL if bad else PASS
    return CheckResult("xodr-04", "planView consistent", status,
                       f"roads with missing/NaN/non-monotonic geometry or length mismatch: {bad[:5]}" if bad else "",
                       len(bad), len(x.roads))


def check_geometry_continuity(x: Xodr) -> CheckResult:
    bad = []
    worst = 0.0
    unsupported = 0
    for r in x.roads:
        geoms = r.findall("planView/geometry")
        for g, g_next in zip(geoms, geoms[1:]):
            end = _geometry_end(g)
            if end is None:
                unsupported += 1
                continue
            gap = math.hypot(end[0] - _f(g_next, "x"), end[1] - _f(g_next, "y"))
            worst = max(worst, gap)
            if gap > CONTINUITY_TOL_M:
                bad.append((r.get("id"), round(gap, 3)))
    total = sum(max(0, len(r.findall("planView/geometry")) - 1) for r in x.roads)
    if unsupported and not bad:
        return CheckResult("xodr-05", "geometry continuity", SKIP, f"{unsupported} unsupported primitives")
    status = FAIL if bad else PASS
    return CheckResult("xodr-05", "geometry continuity", status,
                       f"gaps > {CONTINUITY_TOL_M} m between consecutive primitives (worst {worst:.3f} m): {bad[:5]}"
                       if bad else f"worst gap {worst:.4f} m", len(bad), total)


def check_lanes(x: Xodr) -> CheckResult:
    bad = []
    for r in x.roads:
        sections = r.findall("lanes/laneSection")
        if not sections:
            bad.append((r.get("id"), "no laneSection"))
            continue
        for sec in sections:
            center = sec.findall("center/lane")
            if len(center) != 1 or center[0].get("id") != "0":
                bad.append((r.get("id"), "center lane"))
                break
            left = sorted(int(l.get("id")) for l in sec.findall("left/lane"))
            right = sorted((int(l.get("id")) for l in sec.findall("right/lane")), reverse=True)
            if left != list(range(1, len(left) + 1)) or right != list(range(-1, -len(right) - 1, -1)):
                bad.append((r.get("id"), "lane ids not contiguous"))
                break
            if not left and not right:
                bad.append((r.get("id"), "no driving lanes"))
                break
            road_len = _f(r, "length")
            for lane in sec.findall("left/lane") + sec.findall("right/lane"):
                widths = lane.findall("width")
                if not widths or any(math.isnan(_f(w, "a")) or _f(w, "a") < 0 for w in widths):
                    bad.append((r.get("id"), f"lane {lane.get('id')} width"))
                    break
                offs = [_f(w, "sOffset") for w in widths]
                if offs[0] != 0.0 or any(b <= a for a, b in zip(offs, offs[1:])) or offs[-1] >= road_len + 1e-6:
                    bad.append((r.get("id"), f"lane {lane.get('id')} width sOffset"))
                    break
    status = FAIL if bad else PASS
    return CheckResult("xodr-06", "lane sections", status, f"{bad[:5]}" if bad else "", len(bad), len(x.roads))


def check_road_links(x: Xodr) -> CheckResult:
    bad = []
    total = 0
    for r in x.roads:
        for link in r.findall("link/predecessor") + r.findall("link/successor"):
            total += 1
            etype, eid = link.get("elementType"), link.get("elementId")
            table = x.road_by_id if etype == "road" else x.junction_by_id if etype == "junction" else None
            if table is None or eid not in table:
                bad.append((r.get("id"), link.tag, etype, eid))
    status = FAIL if bad else PASS
    return CheckResult("xodr-07", "road link targets exist", status, f"{bad[:5]}" if bad else "", len(bad), total)


def check_lane_links(x: Xodr) -> CheckResult:
    bad = []
    total = 0
    for r in x.roads:
        lanes = x.lanes_of(r)
        targets = {}
        for link in r.findall("link/predecessor") + r.findall("link/successor"):
            if link.get("elementType") == "road":
                targets[link.tag] = x.road_by_id.get(link.get("elementId"))
        for lane in lanes.values():
            for ll in lane.findall("link/predecessor") + lane.findall("link/successor"):
                total += 1
                target_road = targets.get(ll.tag)
                if target_road is None:
                    continue  # junction-linked: lane links live in the junction
                if ll.get("id") not in x.lanes_of(target_road):
                    bad.append((r.get("id"), lane.get("id"), ll.tag, ll.get("id")))
    status = FAIL if bad else PASS
    return CheckResult("xodr-08", "lane link targets exist", status, f"{bad[:5]}" if bad else "", len(bad), total)


def check_junctions(x: Xodr) -> CheckResult:
    bad = []
    total = 0
    for j in x.junctions:
        conns = j.findall("connection")
        if not conns:
            bad.append((j.get("id"), "no connections"))
            continue
        for c in conns:
            total += 1
            inc, con = x.road_by_id.get(c.get("incomingRoad")), x.road_by_id.get(c.get("connectingRoad"))
            if inc is None or con is None:
                bad.append((j.get("id"), c.get("id"), "road missing"))
                continue
            if con.get("junction") != j.get("id"):
                bad.append((j.get("id"), c.get("id"), f"connecting road {con.get('id')} junction={con.get('junction')}"))
                continue
            inc_lanes, con_lanes = x.lanes_of(inc), x.lanes_of(con)
            for ll in c.findall("laneLink"):
                if ll.get("from") not in inc_lanes or ll.get("to") not in con_lanes:
                    bad.append((j.get("id"), c.get("id"), f"laneLink {ll.get('from')}->{ll.get('to')}"))
                    break
    if not x.junctions:
        return CheckResult("xodr-09", "junction connections", SKIP, "no junctions")
    status = FAIL if bad else PASS
    return CheckResult("xodr-09", "junction connections", status, f"{bad[:5]}" if bad else "", len(bad), total)


def check_junction_roads_referenced(x: Xodr) -> CheckResult:
    """Every road tagged junction=N must appear as a connecting road of junction N."""
    referenced = {(j.get("id"), c.get("connectingRoad")) for j in x.junctions for c in j.findall("connection")}
    bad = [r.get("id") for r in x.roads
           if r.get("junction") not in (None, "-1") and (r.get("junction"), r.get("id")) not in referenced]
    total = sum(1 for r in x.roads if r.get("junction") not in (None, "-1"))
    if not total:
        return CheckResult("xodr-10", "connecting roads referenced", SKIP, "no connecting roads")
    status = FAIL if bad else PASS
    return CheckResult("xodr-10", "connecting roads referenced", status,
                       f"connecting roads missing from their junction: {bad[:5]}" if bad else "", len(bad), total)


def check_numeric_sanity(x: Xodr) -> CheckResult:
    bad = 0
    total = 0
    for el in x.root.iter():
        for k, v in el.attrib.items():
            if k in ("id", "name", "type", "junction", "elementId", "elementType", "contactPoint", "from", "to",
                     "incomingRoad", "connectingRoad", "level", "weight", "color", "date", "version", "revMajor",
                     "revMinor", "subtype", "country", "unit", "dynamic", "orientation", "countryRevision",
                     "pRange", "laneChange", "rule"):
                continue
            try:
                f = float(v)
            except ValueError:
                continue
            total += 1
            if not math.isfinite(f):
                bad += 1
    status = FAIL if bad else PASS
    return CheckResult("xodr-11", "numeric attributes finite", status, f"{bad} NaN/inf values" if bad else "", bad, total)


CHECKS = [
    check_header, check_georeference, check_road_ids_unique, check_planview, check_geometry_continuity,
    check_lanes, check_road_links, check_lane_links, check_junctions, check_junction_roads_referenced,
    check_numeric_sanity,
]


def validate_xodr(x: Xodr) -> list[CheckResult]:
    return [chk(x) for chk in CHECKS]


def validate_file(path) -> list[CheckResult]:
    return validate_xodr(Xodr.parse(path))
