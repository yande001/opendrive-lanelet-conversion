"""Tag unmarked road-edge boundaries as ``type:road_border`` (vm-01-02).

The requirement is that every boundary line (``way``) carry ``type``
(``line_thin``/``line_thick``/``road_border``) and ``lane_change``. odr2cr only
derives a ``type`` from an explicit OpenDRIVE ``roadMark`` — a painted line
(``solid``/``broken`` -> ``line_thin``/``line_thick``) or a ``curb`` -> the
existing ``road_border``. The overwhelming majority of physical road edges are
authored ``roadMark type="none"`` (CARLA/RoadRunner rarely curb-mark them), so
they resolve to ``LineMarking.NO_MARKING`` -> ``unknown`` and crdesigner emits
**no** ``type`` tag at all. The tester's report ("road border lines are not set
to ``type:road_border``") is exactly these untyped outer edges.

Only boundaries of a **drivable** lanelet (``subtype:road`` or
``road_shoulder``) are considered — a ``road_border`` is the edge of the
*carriageway*. A ``walkway``/``crosswalk`` outer edge is the edge of a sidewalk,
not a road border, so it is left untyped (tagging it would also drag sidewalk
corners into the vm-01-05 smoothness scan).

Marking alone cannot tell an unmarked *road edge* from an unmarked *interior
lane divider*, but the lanelet topology can: a drivable boundary is a physical
``road_border`` when it is **not a crossable interior divider between two travel
lanes** — i.e. an outer edge or a boundary with a non-``road`` neighbour
(shoulder / walkway). Those get ``type:road_border`` + ``lane_change:no`` (you
cannot lane-change across the edge of the drivable area).

Two cases are left as-is **by policy** (untyped stays untyped, painted keeps its
marking) because they are crossable dividers between two ``subtype:road``
lanelets, where forcing ``road_border`` would wrongly forbid a legal lane change:

* the divider is one shared way (referenced by two road lanelets); and
* the divider is *coincident but unmerged* — two opposing road lanelets each
  keep their own way for the same physical line (a vm-01-04 modelling defect),
  so each looks like an outer edge by reference count alone. A true outer edge
  has no boundary way coincident with it, so coincidence is the discriminator.

A physical road edge is a ``road_border`` **whatever roadMark the source
painted on it**. odr2cr derives ``type`` from the OpenDRIVE ``roadMark`` — but
that describes the *paint* (``solid``->``line_thin``), a different axis from the
LL2 ``type`` which describes the boundary's *physical function*. Many sources
(the vision-pilot UC-PLN set among them) author a ``solid`` edge line on every
outer road edge, so odr2cr emits ``line_thin`` there and the physical
``road_border`` is lost. So a painted ``line_thin``/``line_thick`` outer edge is
**reclassified** to ``road_border`` (its ``subtype`` dropped); only interior
dividers — which the same topology exempts — keep their painted marking. An
already-physical border type (``road_border``/``curbstone``/``guard_rail``/…) is
left untouched.

Runs as a post-process on the OSM tree, after splitting, so every final
boundary-way piece is classified.
"""
from collections import defaultdict

from lxml import etree

DRIVABLE_SUBTYPES = {"road", "road_shoulder"}   # a road_border edges the carriageway
# Physical-border types already correct — never reclassify these.
PHYSICAL_BORDER_TYPES = {"road_border", "curbstone", "guard_rail", "wall", "fence"}
# Painted lane markings — reclassified to road_border when the topology says the
# way is an outer road edge rather than a crossable interior divider.
PAINTED_TYPES = {"line_thin", "line_thick"}
COINCIDENT_TOL_M = 0.50    # boundary ways within this trace the same physical line
                           # (matches vm-01-04's coincidence tol so no unmerged
                           # opposing centerline is mistagged as an outer edge)
MIN_LEN_M = 1.0            # ignore degenerate sub-metre stubs when testing coincidence


def _tag_map(elem):
    return {t.get("k"): t.get("v") for t in elem.findall("tag")}


def _node_xy(osm_root):
    xy = {}
    for n in osm_root.findall("node"):
        t = _tag_map(n)
        try:
            xy[n.get("id")] = (float(t["local_x"]), float(t["local_y"]))
        except (KeyError, TypeError, ValueError):
            try:
                xy[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))
            except (TypeError, ValueError):
                pass
    return xy


def _polyline(way, xy):
    return [xy[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in xy]


def _len(pts):
    return sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
               for i in range(1, len(pts)))


def _coincident(a, b, tol):
    if len(a) != len(b) or len(a) < 2:
        return False
    fwd = all((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 <= tol * tol
              for i in range(len(a)))
    if fwd:
        return True
    n = len(a)
    return all((a[i][0] - b[n - 1 - i][0]) ** 2 + (a[i][1] - b[n - 1 - i][1]) ** 2 <= tol * tol
               for i in range(n))


def _coincident_boundary_ways(ways, boundary_ids, xy):
    """Ids of boundary ways that trace the same physical line as another boundary way.

    Bucketed by rounded endpoints (1 m) so the pairwise test stays ~O(n); a
    reversed polyline lands in the same bucket.
    """
    polys = {}
    for wid in boundary_ids:
        way = ways.get(wid)
        if way is None:
            continue
        pts = _polyline(way, xy)
        if len(pts) >= 2 and _len(pts) >= MIN_LEN_M:
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
                if _coincident(polys[group[i]], polys[group[j]], COINCIDENT_TOL_M):
                    hit.add(group[i])
                    hit.add(group[j])
    return hit


def _set_road_border(way):
    """Make ``way`` a ``road_border`` in-place (drop painted subtype, set lane_change:no).

    Works for both an untyped way (adds the tags) and a painted
    ``line_thin``/``line_thick`` way (rewrites ``type``, removes the marking
    ``subtype`` which ``road_border`` does not carry).
    """
    have = {}
    for t in way.findall("tag"):
        have[t.get("k")] = t
    if "subtype" in have:                       # road_border carries no subtype
        way.remove(have.pop("subtype"))
    if "type" in have:
        have["type"].set("v", "road_border")
    else:
        etree.SubElement(way, "tag", k="type", v="road_border")
    if "lane_change" in have:
        have["lane_change"].set("v", "no")
    else:
        etree.SubElement(way, "tag", k="lane_change", v="no")


def add_road_borders(osm_root):
    """Tag/reclassify road-edge boundary ways as ``road_border`` in-place.

    Returns a stats dict.
    """
    ways = {w.get("id"): w for w in osm_root.findall("way")}

    # boundary way id -> subtypes of the lanelets that reference it (left/right)
    refs = defaultdict(list)
    for rel in osm_root.findall("relation"):
        tags = _tag_map(rel)
        if tags.get("type") != "lanelet":
            continue
        subtype = tags.get("subtype")
        for mem in rel.findall("member"):
            if mem.get("type") == "way" and mem.get("role") in ("left", "right"):
                refs[mem.get("ref")].append(subtype)

    xy = _node_xy(osm_root)
    coincident = _coincident_boundary_ways(ways, list(refs), xy)

    tagged = added = reclassified = 0
    for wid, subtypes in refs.items():
        way = ways.get(wid)
        if way is None:
            continue
        existing = _tag_map(way).get("type")
        if existing in PHYSICAL_BORDER_TYPES:
            continue        # already a physical border (existing curb / prior border)
        if existing is not None and existing not in PAINTED_TYPES:
            continue        # some other typed line we don't reclassify
        # only the edge of the carriageway is a road_border (skip sidewalk/crosswalk)
        if not any(s in DRIVABLE_SUBTYPES for s in subtypes):
            continue
        # crossable divider between two travel lanes -> keep as-is (painted or untyped)
        if len(subtypes) >= 2 and all(s == "road" for s in subtypes):
            continue
        # coincident-but-unmerged opposing centerline masquerading as an outer edge
        if wid in coincident:
            continue
        _set_road_border(way)
        tagged += 1
        if existing in PAINTED_TYPES:
            reclassified += 1
        else:
            added += 1

    return {"borders_tagged": tagged, "borders_added": added,
            "borders_reclassified": reclassified, "coincident_skipped": len(coincident)}
