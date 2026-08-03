"""Remove duplicate lanelet relations (vm-07-08).

odr2cr can emit several byte-identical lanelet relations for one lane — most
visibly the ~1 m stub where multiple junction connecting-roads share an approach
lane-section, which surfaces as fully-coincident lanelets referencing the *same*
boundary ways. Two lanelets are duplicates iff they carry the **same members**
(role→ref, way and relation alike) and the **same tag set**; connectors that
overlap but differ in ``turn_direction`` are therefore *not* merged — they are a
legal, semantically-distinct maneuver pair (kept, and exempted in the checker).

Runs as a post-process on the OSM tree. Any member in another relation (e.g. a
``right_of_way`` regulatory element) that pointed at a removed lanelet is
repointed to the surviving id, so no dangling reference is left behind.
"""
from collections import defaultdict


def _tag_map(elem):
    return {t.get("k"): t.get("v") for t in elem.findall("tag")}


def _tag_items(elem):
    return tuple(sorted((t.get("k"), t.get("v")) for t in elem.findall("tag")))


def _members(elem):
    return tuple(sorted((m.get("type"), m.get("role"), m.get("ref"))
                        for m in elem.findall("member")))


def dedup_lanelets(osm_root):
    """Collapse identical lanelet relations in-place. Returns stats dict."""
    lanelets = [r for r in osm_root.findall("relation")
                if _tag_map(r).get("type") == "lanelet"]

    groups = defaultdict(list)
    for ll in lanelets:
        mem = _members(ll)
        # need at least one boundary way to be a real, comparable lanelet
        if not any(mtype == "way" for mtype, _, _ in mem):
            continue
        groups[(mem, _tag_items(ll))].append(ll)

    remap = {}          # removed lanelet id -> surviving lanelet id
    removed = dup_groups = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        dup_groups += 1
        keep_id = group[0].get("id")
        for dup in group[1:]:
            remap[dup.get("id")] = keep_id
            osm_root.remove(dup)
            removed += 1

    if remap:
        for rel in osm_root.findall("relation"):
            for mem in rel.findall("member"):
                if mem.get("type") == "relation" and mem.get("ref") in remap:
                    mem.set("ref", remap[mem.get("ref")])

    return {"lanelets_removed": removed, "duplicate_groups": dup_groups}
