"""Trim teleported endpoint nodes from sidewalk/shoulder boundaries.

odr2cr sometimes gives a ``walkway``/``road_shoulder`` lanelet a boundary whose
first and/or last node is a *connecting* node snapped onto a distant lanelet's
node — hundreds of metres away — so the boundary teleports from its real body to
that far point. In a viewer this draws long stray lines shooting across the map
(the "jumping nodes" report).

The defect is unmistakable **before downsampling**: the raw boundary is sampled
sub-metre, so every real segment is tiny and *any* segment longer than
``JUMP_THRESH_M`` is a teleport. Measurement on CARLA Town03 confirms the stray
segment is always at an endpoint (never the interior) and that removing those
endpoints always leaves a healthy body (>=3 nodes). So we simply drop a first/
last node whose adjacent segment exceeds the threshold.

Scope: only ways owned **exclusively** by ``walkway``/``road_shoulder`` lanelets.
A way that any ``road`` (or other) lanelet also uses is left untouched — a road
boundary can legitimately have a long span, and we never risk corrupting it.

Runs pre-downsample (on lat/lon geometry) so dedup/downsample/split all see the
repaired boundaries.
"""
import math

JUMP_THRESH_M = 30.0            # a boundary segment longer than this (pre-downsample) is a teleport
_PED_SUBTYPES = {"walkway", "road_shoulder"}
_R = 6378137.0                  # earth radius (m), matching convert.py's haversine


def _tags(elem):
    return {t.get("k"): t.get("v") for t in elem.findall("tag")}


def _haversine(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * _R * math.asin(math.sqrt(h))


def repair_teleported_endpoints(osm_root, thresh_m=JUMP_THRESH_M):
    """Drop teleported first/last nodes from pedestrian-boundary ways in-place.

    Returns a stats dict. Must run before downsampling (expects lat/lon nodes).
    """
    nodes = {}
    for n in osm_root.findall("node"):
        try:
            nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")))
        except (TypeError, ValueError):
            pass

    # boundary way id -> set of owning lanelet subtypes (via left/right members)
    owners = {}
    for rel in osm_root.findall("relation"):
        if _tags(rel).get("type") != "lanelet":
            continue
        sub = _tags(rel).get("subtype")
        for mem in rel.findall("member"):
            if mem.get("type") == "way" and mem.get("role") in ("left", "right"):
                owners.setdefault(mem.get("ref"), set()).add(sub)

    ways_trimmed = nodes_dropped = 0
    for way in osm_root.findall("way"):
        subs = owners.get(way.get("id"))
        # only boundaries used solely by walkway/road_shoulder lanelets
        if not subs or not subs <= _PED_SUBTYPES:
            continue
        nds = way.findall("nd")
        if len(nds) < 3:
            continue
        coords = [nodes.get(nd.get("ref")) for nd in nds]
        if any(c is None for c in coords):
            continue

        drop_first = _haversine(coords[0], coords[1]) > thresh_m
        drop_last = _haversine(coords[-1], coords[-2]) > thresh_m
        if not (drop_first or drop_last):
            continue
        # never drop below 2 nodes (measurement shows this never triggers, but guard)
        if len(nds) - int(drop_first) - int(drop_last) < 2:
            continue
        if drop_last:
            way.remove(nds[-1])
            nodes_dropped += 1
        if drop_first:
            way.remove(nds[0])
            nodes_dropped += 1
        ways_trimmed += 1

    return {"ways_trimmed": ways_trimmed, "endpoints_dropped": nodes_dropped}
