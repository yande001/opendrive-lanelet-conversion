"""Length-based lanelet splitting (vm-01-24).

Split over-length lanelets into ≤100 m (straight) / ≤20 m (curved) pieces by
cutting their boundary ways at equal arc-length fractions and reassembling the
lanelet relations. Runs on the **pre-downsample** OSM tree, where the geometry
is dense and arc length is accurate; downsampling afterwards only removes
interior nodes, so pieces stay within the limit.

Invariants preserved (the things S4/S6 established):

* **Boundary sharing (vm-01-03/04).** Each way is cut exactly once into an
  ordered list of sub-ways, so every lanelet that referenced it now references
  the same pieces — shared boundaries stay shared. A way is only cut when *all*
  its owning lanelets are splittable, so no relation is left pointing at a
  removed way.
* **Connectivity (vm-01-21).** Consecutive pieces share their cut cross-section
  nodes; the first piece keeps the original start nodes and the last piece the
  original end nodes, so the implicit succ/pred (shared node ids) is preserved.
* **Intersections (vm-03-05).** Lanelets tagged ``turn_direction`` /
  ``intersection_area``, crosswalks, and lanelets referenced by a
  ``right_of_way`` regulatory element are exempt (kept whole); ways they own are
  never cut.

Classification (curved vs straight, length limits) is imported from
``utils.validate`` so the splitter and the checker agree by construction.
"""
import math
from collections import defaultdict

from lxml import etree

from utils.validate import (
    _polyline_len,
    _length_limit,
)

SNAP_M = 0.1  # a cut landing within this of an existing node reuses that node


def _tag_map(elem):
    return {t.get("k"): t.get("v") for t in elem.findall("tag")}


def _node_metric(elem):
    """Metric (x, y) of a node — prefer local_x/local_y, else lon/lat. Mirrors
    the validator so lengths match exactly."""
    t = _tag_map(elem)
    if "local_x" in t and "local_y" in t:
        return float(t["local_x"]), float(t["local_y"])
    return float(elem.get("lon")), float(elem.get("lat"))


def _node_attrs(elem):
    """(lat, lon, local_x, local_y, ele) for interpolating new cut nodes."""
    t = _tag_map(elem)
    return (
        elem.get("lat"),
        elem.get("lon"),
        t.get("local_x"),
        t.get("local_y"),
        t.get("ele"),
    )


def split_long_lanelets(osm_root):
    """Split over-length non-intersection lanelets in place. Returns a small
    stats dict."""
    # ---- parse ------------------------------------------------------------ #
    node_elems = {n.get("id"): n for n in osm_root.findall("node")}
    node_xy = {nid: _node_metric(e) for nid, e in node_elems.items()}

    ways = {}        # wid -> way element
    way_refs = {}    # wid -> [node id]
    way_tags = {}    # wid -> {k: v}
    for w in osm_root.findall("way"):
        wid = w.get("id")
        ways[wid] = w
        way_refs[wid] = [nd.get("ref") for nd in w.findall("nd")]
        way_tags[wid] = _tag_map(w)

    regelem_subtype = {}
    regelems = []    # regulatory_element relation elements (for ref remap)
    lanelets = []    # list of dicts {elem,id,left,right,tags,regmembers}
    for r in osm_root.findall("relation"):
        t = _tag_map(r)
        rtype = t.get("type")
        if rtype == "regulatory_element":
            regelem_subtype[r.get("id")] = t.get("subtype")
            regelems.append(r)
        elif rtype == "lanelet":
            left = right = None
            regmembers = []
            for mem in r.findall("member"):
                role = mem.get("role")
                if role == "left":
                    left = mem.get("ref")
                elif role == "right":
                    right = mem.get("ref")
                elif role == "regulatory_element":
                    regmembers.append(mem.get("ref"))
            lanelets.append({"elem": r, "id": r.get("id"), "left": left,
                             "right": right, "tags": t, "regmembers": regmembers})

    # ---- exemption -------------------------------------------------------- #
    # Intersection connectors (vm-03-05) and crosswalks stay whole. Lanelets
    # referenced by a right_of_way reg-elem are NOT exempt: pinning them would
    # cascade non-splittability through their shared boundaries to the long road
    # lanelets of the same approach. Instead the reg-elem's member refs are
    # remapped to the junction-end piece after splitting (see below).
    #
    def exempt(ll):
        t = ll["tags"]
        return (ll["left"] is None or ll["right"] is None
                or t.get("subtype") == "crosswalk"
                or "turn_direction" in t or "intersection_area" in t)

    is_exempt = {ll["id"]: exempt(ll) for ll in lanelets}

    owners = defaultdict(list)  # way id -> [lanelet dict]
    for ll in lanelets:
        owners[ll["left"]].append(ll)
        owners[ll["right"]].append(ll)

    # ---- per-way length need --------------------------------------------- #
    def way_need(wid):
        pts = [node_xy[r] for r in way_refs[wid] if r in node_xy]
        if len(pts) < 2:
            return 1
        length = _polyline_len(pts)
        limit = _length_limit(pts)
        return math.ceil(length / limit) if length > limit + 1e-9 else 1

    base_need = {wid: way_need(wid) for wid in ways}

    # ---- fixpoint 1: which lanelets are splittable ----------------------- #
    # A way is cuttable only if every owner is splittable; a lanelet is
    # splittable only if both its ways are cuttable. Monotonically shrinks.
    splittable = {ll["id"]: not is_exempt[ll["id"]] for ll in lanelets}
    changed = True
    while changed:
        changed = False
        cuttable = {wid: all(splittable[o["id"]] for o in own)
                    for wid, own in owners.items()}
        for ll in lanelets:
            if is_exempt[ll["id"]]:
                continue
            ok = cuttable.get(ll["left"], False) and cuttable.get(ll["right"], False)
            if splittable[ll["id"]] != ok:
                splittable[ll["id"]] = ok
                changed = True

    sp = [ll for ll in lanelets if splittable[ll["id"]]]

    # ---- fixpoint 2: cut count per way (consistent across a lanelet & shares)
    way_cut = {wid: 1 for wid in ways}
    for ll in sp:
        for wid in (ll["left"], ll["right"]):
            way_cut[wid] = max(way_cut[wid], base_need[wid])
    changed = True
    while changed:
        changed = False
        for ll in sp:
            k = max(way_cut[ll["left"]], way_cut[ll["right"]])
            for wid in (ll["left"], ll["right"]):
                if way_cut[wid] < k:
                    way_cut[wid] = k
                    changed = True

    # ---- id allocation ---------------------------------------------------- #
    max_id = 0
    for elem in osm_root.iter():
        if elem.tag in ("node", "way", "relation"):
            try:
                max_id = max(max_id, int(elem.get("id")))
            except (TypeError, ValueError):
                pass
    counter = [max_id + 1]

    def next_id():
        counter[0] += 1
        return str(counter[0])

    def make_node(a_ref, b_ref, t):
        """Interpolate a new node at fraction t between a_ref and b_ref."""
        a, b = node_elems[a_ref], node_elems[b_ref]
        aa, ba = _node_attrs(a), _node_attrs(b)

        def lerp(x, y):
            if not x or not y:  # None or "" (downsampled nodes have empty lat/lon)
                return None
            return f"{float(x) + t * (float(y) - float(x)):.7f}"

        lat, lon, lx, ly, ele = (lerp(aa[i], ba[i]) for i in range(5))
        nid = next_id()
        node = etree.Element("node", id=nid, visible="true", version="1",
                             lat=lat or "", lon=lon or "")
        if lx is not None:
            node.append(etree.Element("tag", k="local_x", v=lx))
        if ly is not None:
            node.append(etree.Element("tag", k="local_y", v=ly))
        if ele is not None:
            node.append(etree.Element("tag", k="ele", v=ele))
        osm_root.append(node)
        node_elems[nid] = node
        node_xy[nid] = _node_metric(node)
        return nid

    # ---- cut the ways ----------------------------------------------------- #
    def cut_way(wid, k):
        refs = way_refs[wid]
        pts = [node_xy[r] for r in refs]
        seglen = [math.dist(pts[i], pts[i + 1]) for i in range(len(refs) - 1)]
        total = sum(seglen)
        if total <= 0:
            return None
        targets = [total * i / k for i in range(1, k)]
        pieces = []
        cur = [refs[0]]
        acc = 0.0
        ti = 0
        for i in range(len(seglen)):
            seg = seglen[i]
            start, end = acc, acc + seg
            while ti < len(targets) and targets[ti] <= end + 1e-9:
                tgt = targets[ti]
                if abs(tgt - start) <= SNAP_M:
                    cut_ref = refs[i]
                elif abs(tgt - end) <= SNAP_M:
                    cut_ref = refs[i + 1]
                elif seg <= 0:
                    cut_ref = refs[i + 1]
                else:
                    cut_ref = make_node(refs[i], refs[i + 1], (tgt - start) / seg)
                if cut_ref != cur[-1]:
                    cur.append(cut_ref)
                if len(cur) >= 2:
                    pieces.append(cur)
                cur = [cut_ref]
                ti += 1
            if refs[i + 1] != cur[-1]:
                cur.append(refs[i + 1])
            acc = end
        if len(cur) >= 2:
            pieces.append(cur)
        if len(pieces) != k:
            return None  # snapping degenerated; leave this way whole
        # materialise piece ways (copy the original way's tags)
        piece_ids = []
        for piece in pieces:
            pid = next_id()
            pw = etree.Element("way", id=pid, action="modify", visible="true", version="1")
            for ref in piece:
                pw.append(etree.Element("nd", ref=ref))
            for k_, v_ in way_tags[wid].items():
                pw.append(etree.Element("tag", k=k_, v=v_))
            osm_root.append(pw)
            piece_ids.append(pid)
        return piece_ids

    way_pieces = {}
    for wid in list(ways):
        if way_cut[wid] > 1:
            pid_list = cut_way(wid, way_cut[wid])
            if pid_list is not None:
                way_pieces[wid] = pid_list

    # ---- reassemble lanelet relations ------------------------------------ #
    def oriented_pieces(ref_way, other_way):
        """Return other_way's piece ids ordered to match ref_way's stored
        direction (forward unless other_way runs the opposite way)."""
        op = way_pieces[other_way]
        r0 = node_xy[way_refs[ref_way][0]]
        o0 = node_xy[way_refs[other_way][0]]
        o1 = node_xy[way_refs[other_way][-1]]
        if math.dist(o0, r0) <= math.dist(o1, r0):
            return op
        return list(reversed(op))

    n_split = 0
    n_new = 0
    to_remove = []
    end_piece = {}  # old lanelet id -> last piece's relation id (for reg-elem remap)
    for ll in sp:
        left, right = ll["left"], ll["right"]
        if left not in way_pieces or right not in way_pieces:
            continue
        right_seq = way_pieces[right]
        left_seq = oriented_pieces(right, left)
        if len(left_seq) != len(right_seq):
            continue
        k = len(right_seq)
        last = k - 1
        for i in range(k):
            rid = next_id()
            rel = etree.Element("relation", id=rid, action="modify",
                               visible="true", version="1")
            rel.append(etree.Element("member", type="way", ref=right_seq[i], role="right"))
            rel.append(etree.Element("member", type="way", ref=left_seq[i], role="left"))
            for reg in ll["regmembers"]:
                # speed limit applies to every piece; stop/light-type relations
                # only to the final piece (nearest the junction/stop line).
                if regelem_subtype.get(reg) == "speed_limit" or i == last:
                    rel.append(etree.Element("member", type="relation", ref=reg,
                                            role="regulatory_element"))
            for k_, v_ in ll["tags"].items():
                rel.append(etree.Element("tag", k=k_, v=v_))
            osm_root.append(rel)
            if i == last:
                end_piece[ll["id"]] = rid
            n_new += 1
        to_remove.append(ll["elem"])
        n_split += 1

    # A right_of_way reg-elem points at the lanelets that yield / hold priority.
    # When such a lanelet was split, repoint the member to its junction-end piece
    # so the relation still references a live relation id.
    for reg in regelems:
        if regelem_subtype.get(reg.get("id")) != "right_of_way":
            continue
        for mem in reg.findall("member"):
            if mem.get("role") in ("yield", "right_of_way"):
                piece = end_piece.get(mem.get("ref"))
                if piece is not None:
                    mem.set("ref", piece)

    # remove original split relations and cut ways
    for elem in to_remove:
        osm_root.remove(elem)
    for wid in way_pieces:
        osm_root.remove(ways[wid])

    return {"lanelets_split": n_split, "pieces_created": n_new,
            "ways_cut": len(way_pieces)}
