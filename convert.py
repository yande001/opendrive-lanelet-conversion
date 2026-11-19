"""
Convert a single OpenDRIVE (.xodr) file to Lanelet2 (.osm).

Usage:
    python convert.py <input.xodr> [output.osm]

If output path is omitted, writes to ./output/<input_stem>.osm
"""
import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

from lxml import etree
from pyproj import Transformer

# Use the pinned crdesigner submodule (extern/commonroad-scenario-designer).
# Editable-installed into the venv via requirements.txt; this sys.path insert is a
# defensive fallback so the script also works straight from a fresh checkout.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CR_DESIGNER_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "extern", "commonroad-scenario-designer"))
sys.path.insert(0, CR_DESIGNER_PATH)

from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.config.opendrive_config import OpenDriveConfig
from crdesigner.map_conversion.lanelet2.cr2lanelet import CR2LaneletConverter
from commonroad.scenario.scenario import Location, GeoTransformation
from crdesigner.map_conversion.map_conversion_interface import opendrive_to_commonroad

from utils.map_origin import (
    extract_map_origin,
    needs_proj_normalization,
    write_map_origin_yaml,
    write_normalized_xodr,
)
from utils.autoware_config import DEFAULT_CONFIG_PATH, apply_autoware_config
from utils.split import split_long_lanelets
from utils.dedup import dedup_lanelets
from utils.road_border import add_road_borders
from utils.boundary_repair import repair_teleported_endpoints
from utils.odr_s import build_road_reference_index, build_section_index, project_to_s

# --- Constants ---
PROJ_DEG = "EPSG:4326"
PROJ_MET = "EPSG:3857"
R = 6378000  # Earth radius in meters

# Downsampling defaults
DEFAULT_ANGLE_THRSH = 179.9
DEFAULT_MIN_DIST = 3.0
# Assigned-s vs reference-line projection gap (m) above which a mapping row is reported as divergent.
QA_DIVERGENCE_M = 1.0

PointCoords = tuple[float, float]


# --- Conversion ---

def convert_odr_to_osm(input_file, odr_conf=None, map_origin=None):
    """Convert an OpenDRIVE file to a Lanelet2 OSM element tree.

    Returns (osm_root, converter) or (None, None) on failure.
    """
    if odr_conf is None:
        odr_conf = OpenDriveConfig()

    lat = map_origin.latitude if map_origin is not None else 0.0
    lon = map_origin.longitude if map_origin is not None else 0.0
    location = Location(
        geo_name_id=11,
        gps_latitude=lat,
        gps_longitude=lon,
        geo_transformation=GeoTransformation(PROJ_MET),
    )

    scenario = opendrive_to_commonroad(input_file, odr_conf=odr_conf)
    scenario.location = location

    l2conv = CR2LaneletConverter(lanelet2_config)
    osm = l2conv(scenario)
    return osm, l2conv


# --- Downsampling ---

def _coords2xy(p):
    lat, lon = p
    x = math.radians(lon) * R * math.cos(math.radians(lat))
    y = math.radians(lat) * R
    return x, y


def _haversine(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _angle_at(p1, p2, p3):
    a, b, c = _coords2xy(p1), _coords2xy(p2), _coords2xy(p3)
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.degrees(math.acos(max(min(dot / (n1 * n2), 1.0), -1.0)))


def _simplify_indices(points, angle_thrsh, min_dist):
    """Indices of the points kept by downsampling (first and last always kept)."""
    if len(points) <= 2:
        return list(range(len(points)))
    kept = [0]
    last = points[0]
    for i in range(1, len(points) - 1):
        angle = _angle_at(points[i - 1], points[i], points[i + 1])
        dist = _haversine(last, points[i])
        if angle < angle_thrsh and dist >= min_dist:
            kept.append(i)
            last = points[i]
    kept.append(len(points) - 1)
    if len(kept) < 2:
        return [0, len(points) - 1]
    return kept


def _simplify_way(points, angle_thrsh, min_dist):
    return [points[i] for i in _simplify_indices(points, angle_thrsh, min_dist)]


# Ways whose nodes are point features, not lane geometry: never simplify them
# (their nodes carry meaning — e.g. a light_bulbs node's color — and may be
# coincident in 2D, which the simplifier would collapse).
# Area polygons (crosswalk_polygon, intersection_area) are also exempt: every
# corner is meaningful, and the min_dist rule would otherwise drop a corner of a
# narrow quad — a crosswalk_polygon under min_dist wide collapses to a triangle.
_FEATURE_WAY_TYPES = {"light_bulbs", "crosswalk_polygon", "intersection_area"}
# Node tags to carry through the node rebuild (beyond local_x/local_y/ele).
_PRESERVED_NODE_TAGS = ("color", "arrow")


def downsample_osm(osm_root, angle_thrsh=DEFAULT_ANGLE_THRSH, min_dist=DEFAULT_MIN_DIST):
    """Downsample nodes in each way of the OSM tree."""
    transformer = Transformer.from_crs(PROJ_DEG, PROJ_MET, always_xy=True)

    nodes = {}
    node_extra = {}  # id -> {tag: value} for preserved feature tags (e.g. bulb color)
    for node in osm_root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        nodes[node.get("id")] = (
            float(node.get("lat")),
            float(node.get("lon")),
            float(tags.get("ele", 0)),
        )
        extra = {k: tags[k] for k in _PRESERVED_NODE_TAGS if k in tags}
        if extra:
            node_extra[node.get("id")] = extra

    new_node_id_gen = itertools.count(1_000_000)
    # Map each surviving ORIGINAL node id to one new node id. Successive lanelets
    # share their end-cross-section node ids (predecessor's last == successor's
    # first) — connectivity is encoded by that shared identity. Minting a fresh id
    # per (way, point) duplicated those nodes and broke succ/pred routing
    # (vm-01-21), so reuse one new id per original node across every way.
    old_to_new = {}
    new_nodes = {}

    for way in osm_root.findall("way"):
        refs = [nd.get("ref") for nd in way.findall("nd") if nd.get("ref") in nodes]
        if len(refs) < 2:
            continue

        way_type = next((t.get("v") for t in way.findall("tag") if t.get("k") == "type"), None)
        if way_type in _FEATURE_WAY_TYPES:
            kept_refs = refs  # point feature: keep every node, don't simplify
        else:
            coords = [nodes[ref][:2] for ref in refs]
            kept_refs = [refs[i] for i in _simplify_indices(coords, angle_thrsh, min_dist)]
            if len(kept_refs) < 2:
                continue

        for nd in way.findall("nd"):
            way.remove(nd)

        for ref in kept_refs:
            new_id = old_to_new.get(ref)
            if new_id is None:
                lat, lon, ele = nodes[ref]
                local_x, local_y = transformer.transform(lon, lat)
                new_id = str(next(new_node_id_gen))
                old_to_new[ref] = new_id
                node = etree.Element("node", id=new_id, visible="true", version="1", lat="", lon="")
                node.append(etree.Element("tag", k="local_x", v=f"{local_x:.4f}"))
                node.append(etree.Element("tag", k="local_y", v=f"{local_y:.4f}"))
                node.append(etree.Element("tag", k="ele", v=f"{ele:.4f}"))
                for k, v in node_extra.get(ref, {}).items():
                    node.append(etree.Element("tag", k=k, v=v))
                new_nodes[new_id] = node
            way.append(etree.Element("nd", ref=new_id))

    for node in osm_root.findall("node"):
        osm_root.remove(node)
    for node in new_nodes.values():
        osm_root.append(node)

    osm_root.set("generator", "VMB")
    return osm_root


# --- I/O helpers ---

def write_osm(osm_root, path):
    with open(path, "wb") as f:
        f.write(etree.tostring(osm_root, xml_declaration=True, encoding="UTF-8", pretty_print=True))


def _node_xy_table(osm_root):
    """Return ``{node_id: (x, y)}`` for every node of the map, in EPSG:3857 metres.

    The ID mapping compares map nodes with the OpenDRIVE road reference lines
    (see ``utils.odr_s.build_road_reference_index``), so both must use the same
    coordinate system. That system is EPSG:3857 (Web Mercator), the one
    downsampling writes into the ``local_x``/``local_y`` tags.

    Which source is used depends on the node:

    * Nodes with real ``lat``/``lon`` attributes (e.g. when ``--no-downsample``
      was given) are projected from lat/lon to EPSG:3857 here. Their
      ``local_x``/``local_y`` tags, if any, are in a different frame and ignored.
    * Nodes with empty ``lat``/``lon`` (downsampled nodes and nodes created by
      lanelet splitting) already carry EPSG:3857 ``local_x``/``local_y``, which
      are used as-is.

    Nodes with neither are left out.
    """
    to_met = Transformer.from_crs(PROJ_DEG, PROJ_MET, always_xy=True)
    table = {}
    for node in osm_root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if node.get("lat") and node.get("lon"):
            table[node.get("id")] = to_met.transform(float(node.get("lon")), float(node.get("lat")))
        elif "local_x" in tags and "local_y" in tags:
            table[node.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    return table


def _lanelet_chains(osm_root):
    """Find the lanelets that came from each OpenDRIVE lane and put them in driving order.

    One OpenDRIVE lane (within one lane section) usually ends up as several
    lanelets, because long lanelets are cut into shorter pieces
    (``utils.split``). Every lanelet relation carries a temporary ``odr_lane``
    tag ``"<road>.<section>.<lane>"`` naming the lane it came from. This function
    groups the relations by that tag and sorts each group from first piece to
    last, in the direction of travel.

    Sorting uses shared nodes, not coordinates. When a lanelet is cut, the two
    resulting pieces share the nodes at the cut, so piece B directly follows
    piece A exactly when the first node of B's right boundary is the last node
    of A's right boundary. Following these links from the one piece that has no
    predecessor gives the order.

    Returns ``(chains, broken)``:

    * ``chains``: ``{(road, section, lane): [(relation, right_boundary_node_ids), ...]}``,
      pieces in driving order.
    * ``broken``: set of ``(road, section, lane)`` keys whose pieces could not be
      put into one unambiguous sequence. That happens when a piece has no usable
      right boundary, there is no start piece or several, the pieces branch or
      loop, or some pieces are not reachable from the start. These groups are
      not in ``chains``; the caller reports them as skipped instead of guessing
      an order.
    """
    ways = {w.get("id"): w for w in osm_root.findall("way")}
    groups = {}
    broken = set()
    for rel in osm_root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "lanelet" or "odr_lane" not in tags:
            continue
        key = tuple(int(x) for x in tags["odr_lane"].split("."))
        mem = next((m for m in rel.findall("member")
                    if m.get("type") == "way" and m.get("role") == "right"), None)
        way = ways.get(mem.get("ref")) if mem is not None else None
        refs = [nd.get("ref") for nd in way.findall("nd")] if way is not None else []
        if len(refs) < 2:
            broken.add(key)
            refs = []
        groups.setdefault(key, []).append((rel, refs))

    chains = {}
    for key, pieces in groups.items():
        if key in broken:
            continue
        by_first = {}
        for p in pieces:
            by_first.setdefault(p[1][0], []).append(p)
        lasts = {p[1][-1] for p in pieces}
        heads = [p for p in pieces if p[1][0] not in lasts]
        if len(heads) != 1 or any(len(v) != 1 for v in by_first.values()):
            broken.add(key)
            continue
        chain = [heads[0]]
        while len(chain) <= len(pieces):
            nxt = by_first.get(chain[-1][1][-1])
            if nxt is None:
                break
            chain.append(nxt[0])
        if len(chain) != len(pieces):
            broken.add(key)
            continue
        chains[key] = chain
    return chains, broken


def write_id_mapping(osm_root, road_index, section_index, single_side_roads, path):
    """Write the CSV that maps OpenDRIVE lane positions to Lanelet2 lanelet ids.

    Purpose: a scenario written for OpenDRIVE names a position as
    (road id, lane id, ``s``), where ``s`` is the distance along the road's
    reference line. This CSV lets a user find the Lanelet2 lanelet that contains
    that position. Each row describes one lanelet of the output map:

    ======================  =====================================================
    ``opendrive_road_id``   OpenDRIVE road the lanelet came from
    ``opendrive_lane_id``   OpenDRIVE lane id (negative = right of the reference
                            line, positive = left)
    ``opendrive_s_min``     start of the lanelet's stretch along the road's
    ``opendrive_s_max``     reference line (always ``s_min < s_max``)
    ``opendrive_direction`` ``1`` if driving along the lanelet increases ``s``,
                            ``-1`` if it decreases ``s``
    ``lanelet2_relation_id`` id of the lanelet relation in the output ``.osm``
    ======================  =====================================================

    To look up a position, find the row with matching road and lane and
    ``s_min <= s <= s_max``. ``s`` is measured the same way for both driving
    directions, so ``direction`` is only needed to turn ``s`` into a distance
    along the lanelet itself::

        frac = (s - s_min) / (s_max - s_min)
        if direction == -1: frac = 1 - frac
        distance_along_lanelet = frac * lanelet_centerline_length

    It must run on the final map (after dedup and splitting), because those
    steps delete relations and create new ones; only ids that exist in the
    written ``.osm`` may appear in the CSV.

    How the ``s`` ranges are computed, for each OpenDRIVE lane within one lane
    section:

    1. The pieces are put in driving order with ``_lanelet_chains``.
    2. The first piece's start and the last piece's end are set to the exact
       lane-section bounds read from the OpenDRIVE file (``section_index``).
       This makes every road start at exactly ``s = 0`` and leaves no gaps
       between consecutive lane sections.
    3. Each cut between two pieces gets the ``s`` of the shared cut node,
       found by projecting the node onto the road's reference line
       (``road_index``). The two pieces on either side of a cut use the same
       value, so their ranges touch without gaps or overlap.
    4. If a projected value is unusable (no reference line, or it does not lie
       strictly between its neighbours in driving order), that cut falls back
       to splitting the section by the fraction of right-boundary length
       covered so far. This is exact at the section ends but only approximate in
       between: on bends, the length of a lane can differ from the change in
       ``s`` by over 10%. If a fallback value would put the cuts out of order,
       the whole lane uses the fallback values.

    All lengths are measured in the horizontal plane (elevation is ignored),
    because ``s`` is a horizontal distance.

    ``direction`` must describe the lanelet as written, and the upstream
    converter occasionally writes a lanelet's points in reverse order. So when
    the start and end of the lane project far enough apart along the road (at
    least 1 m and a quarter of the section length), the direction is taken from
    the geometry. Otherwise it falls back to the lane id sign, assuming
    right-hand traffic: negative lanes drive towards increasing ``s``. Lanelets
    where geometry and lane sign disagree are counted and reported.

    Quality check: a row is counted as "divergent" when either end of its range
    is more than ``QA_DIVERGENCE_M`` from where the actual lanelet end projects
    onto the reference line. Since cuts normally use the projection directly,
    this mostly flags malformed lanelets.

    Not written, but counted in the returned stats, so they never get a guessed
    row:

    * roads with a ``singleSide`` lane section (their section bounds are not
      reliable, see ``utils.odr_s.build_section_index``);
    * lanes whose pieces could not be ordered (``_lanelet_chains``);
    * lanelets with missing nodes, zero length, lane id 0 or an unknown section.

    With ``--concat``, one lanelet spans several lane sections but its tag names
    only the first, so such lanelets show up as unorderable or divergent.

    Args:
        osm_root: final OSM tree; lanelet relations still carry ``odr_lane`` tags.
        road_index: from ``utils.odr_s.build_road_reference_index``.
        section_index: ``{(road_id, section_idx): (s_start, s_end)}`` from
            ``utils.odr_s.build_section_index``.
        single_side_roads: road ids to skip, from the same function.
        path: output CSV path.

    Returns:
        Stats dict: ``rows`` (written), ``skipped``, ``single_side``,
        ``chain_broken``, ``divergent``, ``worst`` (list of
        ``(gap_m, road, lane, relation_id)``, largest first), ``dir_disagree``,
        ``dir_disagree_roads`` and ``cut_fallback`` (cuts that used the
        length-fraction fallback).
    """
    nodes = _node_xy_table(osm_root)
    chains, broken = _lanelet_chains(osm_root)
    n_rel = sum(len(c) for c in chains.values())
    rows = []
    stats = {"skipped": 0, "single_side": 0, "chain_broken": len(broken), "divergent": 0, "worst": [],
             "dir_disagree": 0, "dir_disagree_roads": set(), "cut_fallback": 0}
    for (road_id, section, lane_id), chain in chains.items():
        if road_id in single_side_roads:
            stats["single_side"] += len(chain)
            continue
        bounds = section_index.get((road_id, section))
        pts = [[nodes.get(r) for r in refs] for _, refs in chain]
        if bounds is None or lane_id == 0 or any(p is None for pl in pts for p in pl):
            stats["skipped"] += len(chain)
            continue
        # Cumulative 2D arc length of the right boundary -> shared fractions f[0]=0 .. f[n]=1.
        cum = [0.0]
        for pl in pts:
            cum.append(cum[-1] + sum(math.dist(a, b) for a, b in zip(pl, pl[1:])))
        if not cum[-1] > 0:
            stats["skipped"] += len(chain)
            continue
        f = [c / cum[-1] for c in cum]
        s_lo, s_hi = bounds
        span = s_hi - s_lo
        n = len(chain)
        ref = road_index.get(road_id)
        # Projection of every cut node (n+1 of them: each piece's first node plus the
        # last piece's last). Cut k is shared by piece k-1 (its end) and piece k (its start).
        cut_nodes = [pl[0] for pl in pts] + [pts[-1][-1]]
        proj = [project_to_s(ref, p) for p in cut_nodes] if ref is not None else None

        # Direction. The lane sign gives the expected value (3.3.1: ids ascend to the
        # left, so negative lanes sit right of the reference line; right-hand traffic
        # runs them with +s and left lanes against it - an LHT map inverts this, and
        # OpenDRIVE only gained an explicit <lane> direction attribute after 1.4). But
        # the column must describe the EMITTED lanelet, and odr2cr occasionally emits
        # one with reversed vertex order. So when the chain's two extreme right-boundary
        # nodes project far enough apart to judge, geometry wins; otherwise (short or
        # degenerate lanelets) fall back to the lane sign. Disagreements are reported.
        expected = 1 if lane_id < 0 else -1
        direction = expected
        if proj is not None and abs(proj[-1] - proj[0]) >= max(1.0, 0.25 * span):
            direction = 1 if proj[-1] > proj[0] else -1
        if direction != expected:
            stats["dir_disagree"] += 1
            stats["dir_disagree_roads"].add(road_id)

        # Cut values in travel order. Fallback: shared arc-length fraction (approximate
        # inside a section on bends, exact at the ends).
        if direction == 1:
            fb = [s_lo + fi * span for fi in f]
        else:                           # travel runs high -> low
            fb = [s_hi - fi * span for fi in f]
        # Ends are pinned analytically (exact road start, sections tile). Interior cuts
        # use the projection of the shared cut node, computed once per cut so the
        # preceding piece's end and the following piece's start are the same number
        # (no gaps). A projected cut must advance strictly in the travel direction and
        # stay strictly inside the pinned ends, else that cut uses the fraction value.
        cuts = [fb[0]] + [None] * (n - 1) + [fb[-1]]
        sign = 1 if direction == 1 else -1
        for k in range(1, n):
            v = proj[k] if proj is not None else None
            if v is not None and (v - cuts[k - 1]) * sign > 0 and (cuts[n] - v) * sign > 0:
                cuts[k] = v
            else:
                cuts[k] = fb[k]
                stats["cut_fallback"] += 1
        if any((cuts[k + 1] - cuts[k]) * sign <= 0 for k in range(n)):
            # A fallback value landed behind an accepted projected cut; use the
            # (monotone by construction) fraction values throughout this chain.
            stats["cut_fallback"] += n - 1
            cuts = list(fb)

        for i, (rel, refs) in enumerate(chain):
            a, b = cuts[i], cuts[i + 1]
            s_min, s_max = min(a, b), max(a, b)
            # QA: assigned s vs projection at both ends of the piece. Pinned ends or
            # fallback cuts that disagree with projection flag malformed lanelets.
            if proj is not None:
                worst = max(abs(proj[i] - a), abs(proj[i + 1] - b))
                if worst > QA_DIVERGENCE_M:
                    stats["divergent"] += 1
                    stats["worst"].append((worst, road_id, lane_id, rel.get("id")))
            rows.append((road_id, lane_id, s_min, s_max, direction, rel.get("id")))
    # Relations inside broken groups were never chained; count them as rows lost.
    stats["skipped"] += sum(
        1 for r in osm_root.findall("relation")
        if any(t.get("k") == "odr_lane" and tuple(int(x) for x in t.get("v").split(".")) in broken
               for t in r.findall("tag")))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opendrive_road_id", "opendrive_lane_id", "opendrive_s_min", "opendrive_s_max",
                    "opendrive_direction", "lanelet2_relation_id"])
        for road_id, lane_id, s0, s1, d, rid in rows:
            w.writerow([road_id, lane_id, f"{s0:.4f}", f"{s1:.4f}", d, rid])
    stats["rows"] = len(rows)
    stats["worst"].sort(reverse=True)
    return stats


def strip_odr_lane_tags(osm_root):
    """Remove the temporary ``odr_lane`` tag from every lanelet relation.

    The upstream converter adds ``odr_lane="<road>.<section>.<lane>"`` so the
    OpenDRIVE origin of each lanelet survives dedup and splitting until
    ``write_id_mapping`` has used it. It is not a Lanelet2/Autoware tag, so it
    is removed before the map is written.
    """
    for rel in osm_root.findall("relation"):
        for t in rel.findall("tag"):
            if t.get("k") == "odr_lane":
                rel.remove(t)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Convert a single OpenDRIVE file to Lanelet2.")
    parser.add_argument("input", help="Path to .xodr file")
    parser.add_argument("output", nargs="?", help="Output .osm path (default: output/<stem>.osm)")
    parser.add_argument("--no-downsample", action="store_true", help="Skip downsampling")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip length-based lanelet splitting (vm-01-24)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip duplicate-lanelet removal (vm-07-08)")
    parser.add_argument("--no-boundary-repair", action="store_true",
                        help="Skip trimming teleported sidewalk/shoulder boundary endpoints")
    parser.add_argument("--no-road-border", action="store_true",
                        help="Skip road_border tagging of unmarked road edges (vm-01-02)")
    parser.add_argument("--no-autoware", action="store_true",
                        help="Disable Autoware-compatible tagging (lane_change, local_x/y, etc.)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Autoware defaults config (YAML); see config/autoware.yaml")
    parser.add_argument("--concat", action="store_true", help="Merge lane sections (default: no merge)")
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_ANGLE_THRSH)
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    stem = input_path.stem
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to output/ so the tool never writes artifacts next to inputs.
        output_path = Path("output") / f"{stem}.osm"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    odr_conf = OpenDriveConfig()
    odr_conf.concatenate_lanelets_flag = args.concat

    # Autoware-compatible tagging (lane_change, one_way:yes/no, ele on every node…).
    lanelet2_config.autoware = not args.no_autoware
    if lanelet2_config.autoware:
        speeds = apply_autoware_config(lanelet2_config, args.config).get("default_speed_kmh")
        if speeds:
            print(f"Autoware config: {args.config} (default speeds {speeds})")

    map_origin = extract_map_origin(input_path)
    origin_path = output_path.with_name(f"map_origin_{stem}.yaml")
    write_map_origin_yaml(origin_path, map_origin)
    if map_origin.source == "proj4":
        print(f"Map origin:     {origin_path} (from <geoReference>)")
    else:
        print(f"Map origin:     {origin_path} (defaults; reason: {map_origin.source})")

    # CARLA's <geoReference> omits `+proj=`, which pyproj rejects. Rewrite to a
    # valid +proj=tmerc string before handing the file to crdesigner.
    conversion_input = input_path
    if needs_proj_normalization(map_origin):
        normalized_path = output_path.with_name(f"normalized_{stem}.xodr")
        write_normalized_xodr(input_path, normalized_path, map_origin)
        conversion_input = normalized_path
        print(f"Normalized:     {normalized_path} (bare <geoReference> rewritten as tmerc)")

    print(f"Converting {input_path} ...")
    osm, converter = convert_odr_to_osm(conversion_input, odr_conf, map_origin=map_origin)
    if osm is None:
        print("Conversion failed.")
        sys.exit(1)

    # Save pre-downsampling
    predown_path = output_path.with_name(f"predown_{stem}.osm")
    write_osm(osm, predown_path)
    print(f"Pre-downsample: {predown_path}")

    # Road reference lines in the OSM metric frame, for the ID mapping written at the
    # end (QA signal only; the s intervals come from the analytic section bounds).
    road_index = build_road_reference_index(conversion_input, converter, odr_conf)

    # Trim teleported endpoint nodes from sidewalk/shoulder boundaries (odr2cr snaps
    # a connecting node onto a far lanelet, drawing long stray "jumping" lines). Runs
    # pre-downsample where the raw sub-metre sampling makes the teleport unambiguous.
    if not args.no_boundary_repair:
        rstats = repair_teleported_endpoints(osm)
        print(f"Boundary repair:trimmed {rstats['endpoints_dropped']} teleported "
              f"endpoints on {rstats['ways_trimmed']} sidewalk/shoulder boundaries")

    # Remove duplicate lanelet relations (vm-07-08). Independent of downsampling —
    # odr2cr emits identical relations (e.g. junction-mouth approach stubs) that
    # would otherwise fully overlap. Kept before splitting so pieces aren't cloned.
    if not args.no_dedup:
        dstats = dedup_lanelets(osm)
        print(f"Dedup:          removed {dstats['lanelets_removed']} duplicate "
              f"lanelets ({dstats['duplicate_groups']} groups)")

    # Downsample
    if not args.no_downsample:
        osm = downsample_osm(osm, args.angle_threshold, args.min_dist)
        node_count = len(osm.findall("node"))
        print(f"Downsampled:    {node_count} nodes")

        # Length-based splitting (vm-01-24) runs after downsampling so it measures
        # the same local_x/local_y geometry the validator does; intersections are
        # exempt (vm-03-05). Needs the metric coords downsampling produces.
        if not args.no_split:
            stats = split_long_lanelets(osm)
            print(f"Split:          {stats['lanelets_split']} lanelets → "
                  f"{stats['pieces_created']} pieces ({stats['ways_cut']} ways cut, "
                  f"{stats['boundaries_decoupled']} walkway boundaries decoupled)")

    # Tag physical road edges as road_border (vm-01-02) — both unmarked edges and
    # painted (line_thin/line_thick) outer edges, which describe paint, not the
    # boundary's physical function. Runs last so every final boundary-way piece
    # (post-split) is classified. Interior dividers between two road lanes keep
    # their marking (or stay untyped) by policy.
    if not args.no_road_border:
        bstats = add_road_borders(osm)
        print(f"Road borders:   tagged {bstats['borders_tagged']} road edges as "
              f"road_border ({bstats['borders_added']} unmarked, "
              f"{bstats['borders_reclassified']} reclassified from painted)")

    # ID mapping, built from the final tree (after dedup/split) so every relation id
    # exists in the output; odr_lane is a helper tag and is stripped right after.
    mapping_path = output_path.with_name(f"id_mapping_{stem}.csv")
    section_index, single_side_roads = build_section_index(conversion_input)
    ms = write_id_mapping(osm, road_index, section_index, single_side_roads, mapping_path)
    print(f"ID mapping:     {mapping_path} ({ms['rows']} entries, {ms['skipped']} skipped, "
          f"{ms['single_side']} singleSide-skipped, {ms['chain_broken']} chain-broken, "
          f"{ms['divergent']} projection-divergent, {ms['cut_fallback']} cuts on arc-length fallback)")
    if ms["dir_disagree"]:
        print(f"  Direction: {ms['dir_disagree']} lanelets run against their lane sign "
              f"(roads {sorted(ms['dir_disagree_roads'])}); geometry used")
    for worst, road_id, lane_id, rid in ms["worst"][:5]:
        print(f"  QA divergent: road {road_id} lane {lane_id} relation {rid} ({worst:.1f} m)")
    strip_odr_lane_tags(osm)

    write_osm(osm, output_path)
    print(f"Output:         {output_path}")


if __name__ == "__main__":
    main()
