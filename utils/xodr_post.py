"""Post-processing of the OpenDRIVE XML produced by crdesigner's cr2odr writer.

cr2odr synthesizes one <junction> per *incoming road* that has several successor roads.
At a real intersection several incoming roads share the same connecting roads, so the
same connecting road ends up listed in two or three junctions while its ``junction``
attribute can only name one of them. OpenDRIVE (and CARLA) expect one junction per
intersection. ``merge_junctions`` unions junctions that share a connecting road, moves
all their connections into the surviving junction and rewrites every reference.
"""

from __future__ import annotations

from collections import defaultdict

from lxml import etree


def _find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def merge_junctions(root: etree._Element) -> dict:
    """Merge synthesized junctions that share connecting roads. Mutates ``root`` in place."""
    junctions = root.findall("junction")
    by_id = {j.get("id"): j for j in junctions}
    parent = {jid: jid for jid in by_id}

    owner: dict[str, str] = {}
    for j in junctions:
        for c in j.findall("connection"):
            road = c.get("connectingRoad")
            if road in owner:
                a, b = _find(parent, owner[road]), _find(parent, j.get("id"))
                if a != b:
                    # keep the numerically smallest id as the representative
                    lo, hi = sorted((a, b), key=lambda s: (len(s), s))
                    parent[hi] = lo
            else:
                owner[road] = j.get("id")

    groups: dict[str, list[str]] = defaultdict(list)
    for jid in by_id:
        groups[_find(parent, jid)].append(jid)

    remap: dict[str, str] = {}
    merged_away = 0
    dropped_duplicates = 0
    for rep, members in groups.items():
        if len(members) == 1:
            continue
        target = by_id[rep]
        seen = {(c.get("incomingRoad"), c.get("connectingRoad")) for c in target.findall("connection")}
        for jid in members:
            if jid == rep:
                continue
            remap[jid] = rep
            src = by_id[jid]
            for c in list(src.findall("connection")):
                key = (c.get("incomingRoad"), c.get("connectingRoad"))
                src.remove(c)
                if key in seen:
                    dropped_duplicates += 1
                    continue
                seen.add(key)
                target.append(c)
            src.getparent().remove(src)
            merged_away += 1
        for i, c in enumerate(target.findall("connection")):
            c.set("id", str(i))

    # rewrite references
    rewritten = 0
    for road in root.findall("road"):
        j = road.get("junction")
        if j in remap:
            road.set("junction", remap[j])
            rewritten += 1
        for link in road.findall("link/predecessor") + road.findall("link/successor"):
            if link.get("elementType") == "junction" and link.get("elementId") in remap:
                link.set("elementId", remap[link.get("elementId")])
                rewritten += 1

    # the surviving junction must own every connecting road listed in it
    fixed_owner = 0
    road_by_id = {r.get("id"): r for r in root.findall("road")}
    for j in root.findall("junction"):
        for c in j.findall("connection"):
            r = road_by_id.get(c.get("connectingRoad"))
            if r is not None and r.get("junction") != j.get("id"):
                r.set("junction", j.get("id"))
                fixed_owner += 1

    return {"junctions_before": len(junctions), "junctions_after": len(root.findall("junction")),
            "merged_away": merged_away, "duplicate_connections_dropped": dropped_duplicates,
            "references_rewritten": rewritten, "connecting_road_owner_fixed": fixed_owner}


def postprocess_file(path, merge: bool = True) -> dict:
    """Apply the post-processing to an .xodr file in place; returns the stats."""
    tree = etree.parse(str(path))
    stats = merge_junctions(tree.getroot()) if merge else {}
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return stats
