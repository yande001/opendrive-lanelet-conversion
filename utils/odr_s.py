"""OpenDRIVE road geometry needed by the OpenDRIVE -> Lanelet2 ID mapping.

In OpenDRIVE, a position along a road is given by ``s``: the distance along the
road's reference line from the road's start. The output Lanelet2 map has no
``s`` values, so the ID mapping (``convert.write_id_mapping``) needs two things
from the original ``.xodr`` file to assign each lanelet an ``s`` range:

* ``build_section_index``: the exact ``s`` range of every lane section, read
  directly from the file. Used for the start and end of each lane section.
* ``build_road_reference_index`` + ``project_to_s``: each road's reference line
  as a polyline in the output map's coordinate system, so the ``s`` of any map
  point can be found by projecting it onto that line. Used where lanelets were
  cut into pieces inside a lane section.

For projection to work, the reference line must go through exactly the same
coordinate transforms the converter applies to the lanelets, in the same order:

1. plan-view (x, y) from the parser;
2. the OpenDRIVE projection (header ``geoReference`` -> ``proj_string_odr``), as
   ``network.export_commonroad_scenario`` applies to every lanelet;
3. the ``converter.origin_utm`` offset, then ``converter.transformer`` to
   lat/lon, as ``cr2lanelet._create_nodes_from_vertices`` does;
4. lat/lon -> EPSG:3857 with ``always_xy``, as ``downsample_osm`` writes
   ``local_x``/``local_y``.

If step 2 is skipped, the reference line stays in the raw ``.xodr`` frame, far
from the lanelets, and every projection silently lands on a road endpoint.
"""

from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

PROJ_DEG = "EPSG:4326"
PROJ_MET = "EPSG:3857"

# Reference-line sampling step. Projection error on a curve is bounded by the
# chord sagitta, which at 1 m is far below the ~0.5 m tolerance we care about.
SAMPLE_STEP = 1.0


def build_road_reference_index(xodr_path, converter, odr_conf, step=SAMPLE_STEP):
    """Sample every road's reference line into the output map's coordinate system.

    Each road's reference line is sampled every ``step`` metres of ``s`` (plus
    its exact end) and transformed to EPSG:3857 metres, the system of the map's
    ``local_x``/``local_y`` tags (see the module docstring for the transform
    chain). The result is what ``project_to_s`` uses to find the ``s`` of a map
    point.

    Args:
        xodr_path: the ``.xodr`` file the converter actually read. If the
            converter was given a rewritten copy (e.g. with a corrected
            ``geoReference``), pass that copy, not the original.
        converter: the ``CR2LaneletConverter`` used for the conversion; its
            ``origin_utm`` and ``transformer`` define the output frame.
        odr_conf: the ``OpenDriveConfig`` used for the conversion.
        step: sampling interval in metres of ``s``.

    Returns:
        ``{road_id: (points, s_values, length)}``: ``points`` is an (N, 2) array
        of EPSG:3857 coordinates, ``s_values`` the (N,) ``s`` of each point, and
        ``length`` the road length. Roads of zero length are left out.
    """
    from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser import parse_opendrive

    odr = parse_opendrive(xodr_path)
    # Same projection odr2cr applies to the lanelets; identity when the file has no
    # geoReference or the config disables it (network.export_commonroad_scenario).
    geo_ref = odr.header.geo_reference
    if geo_ref is not None and odr_conf.proj_string_odr is not None:
        odr_proj = Transformer.from_proj(CRS(geo_ref), CRS(odr_conf.proj_string_odr))
    else:
        odr_proj = None
    to_met = Transformer.from_crs(PROJ_DEG, PROJ_MET, always_xy=True)
    ox, oy = converter.origin_utm

    index = {}
    for road in odr.roads:
        length = float(road.plan_view.length)
        if not length > 0:
            continue
        # Sample every `step` metres plus the exact end so cum_s spans [0, length].
        s_vals = list(np.arange(0.0, length, step)) + [length]
        pts = []
        for s in s_vals:
            pos, _, _, _ = road.plan_view.calc(float(s), compute_curvature=False)
            x, y = odr_proj.transform(*pos) if odr_proj is not None else pos
            lat, lon = converter.transformer.transform(ox + x, oy + y)
            pts.append(to_met.transform(lon, lat))
        index[int(road.id)] = (np.asarray(pts, dtype=float), np.asarray(s_vals, dtype=float), length)
    return index


def project_to_s(entry, point):
    """Return the ``s`` of the reference-line point closest to ``point``.

    Finds the closest point on the sampled reference line ``entry`` (one value
    of ``build_road_reference_index``) to ``point`` (x, y in EPSG:3857) and
    returns its ``s``, interpolated between the two neighbouring samples.
    The result is limited to ``[0, road length]``, so a point beyond either end
    of the road gets that end's ``s``.
    """
    pts, cum_s, length = entry
    p = np.asarray(point, dtype=float)
    a = pts[:-1]
    d = pts[1:] - a
    seg_len2 = np.einsum("ij,ij->i", d, d)
    # Zero-length segments (repeated samples) get t=0 instead of dividing by zero.
    t = np.where(seg_len2 > 0, np.einsum("ij,ij->i", p - a, d) / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = a + d * t[:, None]
    i = int(np.argmin(np.einsum("ij,ij->i", p - foot, p - foot)))
    s = cum_s[i] + t[i] * (cum_s[i + 1] - cum_s[i])
    return float(min(max(s, 0.0), length))


def build_section_index(xodr_path):
    """Read the exact ``s`` range of every lane section from the ``.xodr`` file.

    An OpenDRIVE road is divided along its length into lane sections; each
    starts at an ``s`` given in the file (``sPos``) and ends where the next one
    starts, or at the road's end. The standard requires the first section to
    start at ``s = 0`` and the sections to be in increasing order, so together
    they cover the whole road without gaps. These bounds are exact, unlike
    values recovered from map geometry.

    Some roads use ``<laneSection singleSide="true">`` (OpenDRIVE 1.4), meaning
    the section applies to only one side of the road and the two sides change
    sections at different ``s``. The parser ignores this and computes one range
    for both sides, which is wrong for one of them. Those roads are returned
    separately so callers can skip them instead of using wrong ranges.

    Returns:
        ``(sections, single_side_roads)``:

        * ``sections``: ``{(road_id, section_idx): (s_start, s_end)}``.
          ``section_idx`` is the same index used in the ``odr_lane`` tag
          (``"<road>.<section>.<lane>"``).
        * ``single_side_roads``: set of road ids with any ``singleSide`` section.
    """
    from crdesigner.map_conversion.opendrive.odr2cr.opendrive_parser.parser import parse_opendrive

    odr = parse_opendrive(Path(xodr_path))
    sections = {}
    single_side = set()
    for road in odr.roads:
        for sec in road.lanes.lane_sections:
            # The parser stores the raw attribute string (None when absent).
            if str(sec.single_side).strip().lower() in ("true", "1"):
                single_side.add(int(road.id))
            sections[(int(road.id), int(sec.idx))] = (float(sec.sPos), float(sec.sPos) + float(sec.length))
    return sections, single_side
