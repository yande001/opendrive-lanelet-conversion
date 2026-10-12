"""Round-trip fidelity check for Lanelet2 -> OpenDRIVE conversions.

The generated ``.xodr`` is read back with crdesigner's OpenDRIVE reader
(``opendrive_to_commonroad``) and its lanelets are compared, as polygons, with the lanelets
of the original ``.osm`` (read with the same Lanelet2 reader and projection the pipeline
used). Metrics: union IoU, per-lanelet best-match IoU, boundary deviation percentiles.
Optionally saves an overlay plot (original grey, round-trip red, worst matches annotated).

Both reads go through crdesigner so systematic reader behaviour cancels out; what remains
is the loss introduced by the OpenDRIVE geometry model (line/arc/spiral fit, linear lane
widths, one lane section per road).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from crdesigner.common.config.general_config import general_config
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.config.opendrive_config import OpenDriveConfig
from crdesigner.map_conversion.map_conversion_interface import (
    lanelet_to_commonroad,
    opendrive_to_commonroad,
)

from utils.cr_prep import strip_z

BOUNDARY_SAMPLE_STEP_M = 0.5


@dataclass
class RoundtripMetrics:
    lanelets_original: int
    lanelets_roundtrip: int
    area_original_m2: float
    area_roundtrip_m2: float
    union_iou: float
    uncovered_area_m2: float      # original area not covered by the round-trip
    extra_area_m2: float          # round-trip area outside the original
    match_iou_p50: float          # per original lanelet: best-matching round-trip lanelet
    match_iou_p90: float          # (p90 here = 10th percentile, i.e. the bad tail)
    match_iou_min: float
    boundary_dev_p50_m: float     # original boundary samples -> nearest round-trip boundary
    boundary_dev_p90_m: float
    boundary_dev_max_m: float
    worst: list                   # [(osm relation id or cr id, best iou), ...]

    def summary(self) -> dict:
        return {
            "lanelets": f"{self.lanelets_original}->{self.lanelets_roundtrip}",
            "union_iou": round(self.union_iou, 3),
            "uncovered_m2": round(self.uncovered_area_m2, 1),
            "extra_m2": round(self.extra_area_m2, 1),
            "match_iou_p50/p90/min": f"{self.match_iou_p50:.3f}/{self.match_iou_p90:.3f}/{self.match_iou_min:.3f}",
            "boundary_dev_p50/p90/max_m": f"{self.boundary_dev_p50_m:.2f}/{self.boundary_dev_p90_m:.2f}/{self.boundary_dev_max_m:.2f}",
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _lanelet_polygon(la) -> Polygon:
    pts = np.vstack([la.left_vertices[:, :2], la.right_vertices[::-1, :2]])
    poly = Polygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def load_original(osm_path: str | Path, proj_string: str):
    """Original lanelets as (label, polygon, left, right) under ``proj_string``."""
    general_config.proj_string_cr = proj_string
    lanelet2_config.translate = False
    scenario = lanelet_to_commonroad(str(osm_path), general_conf=general_config, lanelet2_conf=lanelet2_config)
    old_ids = getattr(scenario.lanelet_network, "_old_lanelet_ids", {})
    cr_to_osm = {v: k for k, v in old_ids.items()}
    strip_z(scenario)
    out = []
    for la in scenario.lanelet_network.lanelets:
        out.append((str(cr_to_osm.get(la.lanelet_id, la.lanelet_id)), _lanelet_polygon(la),
                    la.left_vertices[:, :2], la.right_vertices[:, :2]))
    return out


def load_roundtrip(xodr_path: str | Path):
    """Lanelets of the .xodr in the file's own coordinates (no re-projection)."""
    conf = OpenDriveConfig()
    saved = conf.proj_string_odr
    try:
        # class-level descriptor: None disables the geoReference transform entirely
        conf.proj_string_odr = None
        conf.concatenate_lanelets_flag = False
        scenario = opendrive_to_commonroad(str(xodr_path), odr_conf=conf)
    finally:
        conf.proj_string_odr = saved
    return [(str(la.lanelet_id), _lanelet_polygon(la), la.left_vertices[:, :2], la.right_vertices[:, :2])
            for la in scenario.lanelet_network.lanelets]


def _sample_boundary(poly: np.ndarray, step: float) -> np.ndarray:
    line = LineString(poly)
    if line.length == 0:
        return poly[:1]
    d = np.arange(0.0, line.length, step)
    pts = [line.interpolate(x) for x in d] + [Point(poly[-1])]
    return np.array([(p.x, p.y) for p in pts])


def roundtrip_compare(osm_path: str | Path, xodr_path: str | Path, proj_string: str,
                      png_path: Optional[str | Path] = None,
                      boundary_step: float = BOUNDARY_SAMPLE_STEP_M) -> RoundtripMetrics:
    orig = load_original(osm_path, proj_string)
    rt = load_roundtrip(xodr_path)

    orig_u = unary_union([p for _, p, _, _ in orig])
    rt_u = unary_union([p for _, p, _, _ in rt])
    inter = orig_u.intersection(rt_u).area
    union = orig_u.union(rt_u).area

    # per-lanelet best match
    rt_polys = [p for _, p, _, _ in rt]
    tree = STRtree(rt_polys) if rt_polys else None
    best = []
    for label, poly, _, _ in orig:
        iou = 0.0
        if tree is not None:
            for idx in tree.query(poly):
                cand = rt_polys[idx]
                u = poly.union(cand).area
                if u > 0:
                    iou = max(iou, poly.intersection(cand).area / u)
        best.append((label, iou))
    ious = np.array([b[1] for b in best]) if best else np.array([0.0])

    # boundary deviation: original boundary samples vs nearest round-trip boundary line
    rt_lines = [LineString(l) for _, _, l, _ in rt] + [LineString(r) for _, _, _, r in rt]
    line_tree = STRtree(rt_lines) if rt_lines else None
    devs = []
    for _, _, left, right in orig:
        for side in (left, right):
            for p in _sample_boundary(side, boundary_step):
                pt = Point(p)
                if line_tree is None:
                    devs.append(float("inf"))
                    continue
                idx = line_tree.nearest(pt)
                devs.append(rt_lines[idx].distance(pt))
    devs = np.array(devs) if devs else np.array([0.0])

    worst = sorted(best, key=lambda b: b[1])[:10]
    metrics = RoundtripMetrics(
        lanelets_original=len(orig), lanelets_roundtrip=len(rt),
        area_original_m2=orig_u.area, area_roundtrip_m2=rt_u.area,
        union_iou=(inter / union) if union > 0 else 0.0,
        uncovered_area_m2=orig_u.difference(rt_u).area, extra_area_m2=rt_u.difference(orig_u).area,
        match_iou_p50=float(np.percentile(ious, 50)), match_iou_p90=float(np.percentile(ious, 10)),
        match_iou_min=float(ious.min()),
        boundary_dev_p50_m=float(np.percentile(devs, 50)), boundary_dev_p90_m=float(np.percentile(devs, 90)),
        boundary_dev_max_m=float(devs.max()), worst=[(l, round(i, 3)) for l, i in worst],
    )
    if png_path:
        _plot(orig, rt, worst, metrics, png_path)
    return metrics


def _plot(orig, rt, worst, metrics: RoundtripMetrics, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rings(poly):
        return list(poly.geoms) if hasattr(poly, "geoms") else [poly]

    fig, ax = plt.subplots(figsize=(18, 10), dpi=110)
    for _, poly, _, _ in orig:
        for g in rings(poly):
            ax.fill(*g.exterior.xy, color="0.75", alpha=0.6, lw=0)
            ax.plot(*g.exterior.xy, color="0.4", lw=0.5)
    for _, poly, _, _ in rt:
        for g in rings(poly):
            ax.plot(*g.exterior.xy, color="tab:red", lw=0.6, alpha=0.9)
    by_label = {label: poly for label, poly, _, _ in orig}
    for label, iou in worst:
        c = by_label[label].centroid
        ax.annotate(f"{label}\n{iou:.2f}", (c.x, c.y), fontsize=6, color="tab:blue", ha="center")
    ax.set_aspect("equal")
    ax.set_title(f"grey = original Lanelet2, red = OpenDRIVE round-trip | union IoU {metrics.union_iou:.3f}, "
                 f"boundary dev p90 {metrics.boundary_dev_p90_m:.2f} m, max {metrics.boundary_dev_max_m:.2f} m")
    fig.savefig(str(png_path), bbox_inches="tight")
    plt.close(fig)
