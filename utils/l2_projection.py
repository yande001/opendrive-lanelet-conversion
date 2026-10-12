"""Projection handling for the Lanelet2 (.osm) -> OpenDRIVE (.xodr) direction.

Autoware Lanelet2 maps carry WGS-84 ``lat``/``lon`` on every node plus the metric
``local_x``/``local_y`` tags the stack actually drives on. crdesigner's Lanelet2 reader only
looks at ``lat``/``lon`` and projects them with ``general_config.proj_string_cr``, so the
whole point of this module is to pick a PROJ string under which

    proj(lat, lon) == (local_x, local_y)

(the "mgrs" frame), or a map-centred transverse Mercator (the "local" frame, coordinates
near the origin and a ``geoReference`` whose ``+lat_0``/``+lon_0`` consumers such as CARLA
read for GNSS). The chosen string is verified against the nodes, never trusted blindly.

Resolution order (first hit wins): explicit ``--proj`` > ``map_projector_info.yaml`` >
least-squares fit of ``local_x/local_y`` against UTM / Web-Mercator > UTM zone of the mean
longitude (warned).

See docs/arch/l2_to_odr.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml
from lxml import etree
from pyproj import CRS, Transformer

from utils.map_origin import DEFAULT_SCALE_FACTOR, MapOrigin, _utm_central_meridian

# Autoware map_projector_info.yaml projector types
PROJECTOR_MGRS = "MGRS"
PROJECTOR_TMERC = "TransverseMercator"
PROJECTOR_LOCAL_UTM = "LocalCartesianUTM"
PROJECTOR_LOCAL = "local"

FRAME_MGRS = "mgrs"      # x/y == Autoware local_x/local_y
FRAME_LOCAL = "local"    # x/y near the origin, tmerc centred on the map

# Node-check tolerance: anything worse than this means the proj string is wrong,
# not just numerically noisy (local_x/local_y are written with 4 decimals).
NODE_CHECK_WARN_M = 0.01
FIT_ACCEPT_RMSE_M = 0.05

# A *valid* proj4 form of EPSG:3857 that accepts +x_0/+y_0 offsets (the EPSG code does not).
_WEB_MERCATOR_PROJ4 = (
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +k=1 +units=m +nadgrids=@null +no_defs"
)

NodeSample = tuple[float, float, float, float]  # lat, lon, local_x, local_y


@dataclass
class ProjInfo:
    """Everything the pipeline and its sidecars need to know about the projection."""

    proj_string: str                      # string used for the conversion / geoReference
    source: str                           # cli | projector_info-<type> | fit-<crs> | fallback-utm
    frame: str = FRAME_MGRS
    native_proj_string: Optional[str] = None  # string reproducing local_x/local_y (mgrs frame)
    projector_type: Optional[str] = None
    mgrs_grid: Optional[str] = None
    origin_lat: Optional[float] = None    # +lat_0 of proj_string
    origin_lon: Optional[float] = None    # +lon_0 of proj_string
    scale_factor: Optional[float] = None  # +k of proj_string
    fit_rmse_m: Optional[float] = None
    max_node_error_m: Optional[float] = None
    nodes_checked: int = 0
    restored_latlon: int = 0
    notes: Optional[str] = None

    def to_map_origin(self) -> MapOrigin:
        """Autoware map_loader view of this projection (only exact for the local frame)."""
        return MapOrigin(
            projector_type=PROJECTOR_TMERC,
            vertical_datum="WGS84",
            latitude=self.origin_lat or 0.0,
            longitude=self.origin_lon or 0.0,
            scale_factor=self.scale_factor if self.scale_factor is not None else DEFAULT_SCALE_FACTOR,
            source=self.source,
            proj_string=self.proj_string,
        )


# --- helpers -----------------------------------------------------------------------------

def _tmerc(lat_0: float, lon_0: float, k: float, x_0: float, y_0: float) -> str:
    return (
        f"+proj=tmerc +lat_0={lat_0:.10g} +lon_0={lon_0:.10g} +k={k:.10g} "
        f"+x_0={x_0:.10g} +y_0={y_0:.10g} +datum=WGS84 +units=m +no_defs"
    )


def _utm_zone(lon: float) -> int:
    return int(math.floor((lon + 180.0) / 6.0)) + 1


def _utm_tmerc(zone: int, south: bool, dx: float = 0.0, dy: float = 0.0) -> str:
    """UTM written out as tmerc so the false easting/northing can absorb an offset."""
    return _tmerc(0.0, _utm_central_meridian(zone), DEFAULT_SCALE_FACTOR,
                  500000.0 + dx, (10_000_000.0 if south else 0.0) + dy)


def _proj_param(proj_string: str, key: str) -> Optional[float]:
    m = re.search(rf"\+{key}=([^\s]+)", proj_string)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _to_latlon_transformer(proj_string: str) -> Transformer:
    """lat/lon (EPSG:4326, always lat-first) -> projected x/y."""
    return Transformer.from_crs(CRS("EPSG:4326"), CRS(proj_string), always_xy=False)


def node_samples(osm_root: etree._Element, max_nodes: int = 2000) -> list[NodeSample]:
    """Nodes that have both lat/lon and local_x/local_y, evenly subsampled."""
    nodes = osm_root.findall("node")
    if not nodes:
        return []
    step = max(1, len(nodes) // max_nodes)
    out: list[NodeSample] = []
    for n in nodes[::step]:
        lat, lon = n.get("lat"), n.get("lon")
        lx = n.find("tag[@k='local_x']")
        ly = n.find("tag[@k='local_y']")
        if not lat or not lon or lx is None or ly is None:
            continue
        try:
            out.append((float(lat), float(lon), float(lx.get("v")), float(ly.get("v"))))
        except ValueError:
            continue
    return out


def node_latlon(osm_root: etree._Element) -> np.ndarray:
    vals = []
    for n in osm_root.findall("node"):
        lat, lon = n.get("lat"), n.get("lon")
        if lat and lon:
            try:
                vals.append((float(lat), float(lon)))
            except ValueError:
                pass
    return np.array(vals) if vals else np.empty((0, 2))


def verify_proj(proj_string: str, samples: Sequence[NodeSample]) -> tuple[float, int]:
    """Max |proj(lat,lon) - (local_x, local_y)| over the samples, and the sample count."""
    if not samples:
        return float("nan"), 0
    tr = _to_latlon_transformer(proj_string)
    arr = np.asarray(samples)
    x, y = tr.transform(arr[:, 0], arr[:, 1])
    err = np.hypot(np.asarray(x) - arr[:, 2], np.asarray(y) - arr[:, 3])
    return float(np.max(err)), len(samples)


# --- sources -----------------------------------------------------------------------------

def read_projector_info(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
    return data


def proj_from_mgrs_grid(grid: str) -> ProjInfo:
    """MGRS 100 km grid square -> tmerc whose x/y equal Autoware's local_x/local_y.

    Autoware's MGRS projector returns the position inside the 100 km square, i.e.
    UTM easting/northing minus the square's south-west corner. ``51RUH`` ->
    zone 51 N, corner (300000, 2700000), so x_0 = 500000 - 300000 = 200000 and
    y_0 = -2700000.
    """
    import mgrs  # crdesigner dependency; imported lazily so the module loads without it

    zone, hemisphere, easting0, northing0 = mgrs.MGRS().MGRSToUTM(grid.strip())
    south = str(hemisphere).upper().startswith("S")
    proj = _utm_tmerc(int(zone), south, dx=-float(easting0), dy=-float(northing0))
    return ProjInfo(
        proj_string=proj,
        native_proj_string=proj,
        source="projector_info-MGRS",
        projector_type=PROJECTOR_MGRS,
        mgrs_grid=grid.strip(),
        origin_lat=0.0,
        origin_lon=float(_utm_central_meridian(int(zone))),
        scale_factor=DEFAULT_SCALE_FACTOR,
    )


def proj_from_projector_info(info: dict) -> Optional[ProjInfo]:
    """Translate an Autoware ``map_projector_info.yaml`` into a ProjInfo (None if local)."""
    ptype = str(info.get("projector_type", "")).strip()
    origin = info.get("map_origin") or {}
    lat = origin.get("latitude")
    lon = origin.get("longitude")

    if ptype == PROJECTOR_MGRS:
        grid = info.get("mgrs_grid")
        if not grid:
            raise ValueError("projector_type MGRS but no mgrs_grid in map_projector_info.yaml")
        return proj_from_mgrs_grid(str(grid))

    if ptype == PROJECTOR_TMERC:
        if lat is None or lon is None:
            raise ValueError("projector_type TransverseMercator needs map_origin latitude/longitude")
        k = float(info.get("scale_factor", DEFAULT_SCALE_FACTOR))
        proj = _tmerc(float(lat), float(lon), k, 0.0, 0.0)
        return ProjInfo(proj_string=proj, native_proj_string=proj,
                        source="projector_info-TransverseMercator", projector_type=ptype,
                        origin_lat=float(lat), origin_lon=float(lon), scale_factor=k)

    if ptype == PROJECTOR_LOCAL_UTM:
        # Autoware: UTM(zone of origin) minus UTM(origin). Written as tmerc with the
        # projected origin folded into the false easting/northing. UNTESTED: no sample map.
        if lat is None or lon is None:
            raise ValueError("projector_type LocalCartesianUTM needs map_origin latitude/longitude")
        zone = _utm_zone(float(lon))
        south = float(lat) < 0
        base = _utm_tmerc(zone, south)
        ex, ny = _to_latlon_transformer(base).transform(float(lat), float(lon))
        # subtract the origin's UTM easting/northing via the false easting/northing
        proj = _tmerc(0.0, _utm_central_meridian(zone), DEFAULT_SCALE_FACTOR,
                      500000.0 - float(ex), (10_000_000.0 if south else 0.0) - float(ny))
        return ProjInfo(proj_string=proj, native_proj_string=proj,
                        source="projector_info-LocalCartesianUTM", projector_type=ptype,
                        origin_lat=0.0, origin_lon=float(_utm_central_meridian(zone)),
                        scale_factor=DEFAULT_SCALE_FACTOR,
                        notes="LocalCartesianUTM branch is untested (no sample map)")

    if ptype in (PROJECTOR_LOCAL, ""):
        return None
    raise ValueError(f"Unknown projector_type {ptype!r} in map_projector_info.yaml")


def proj_from_node_fit(samples: Sequence[NodeSample]) -> Optional[ProjInfo]:
    """Find a CRS + translation reproducing local_x/local_y from lat/lon.

    Tries UTM (zone of the mean longitude and its neighbours) and Web Mercator (this
    repo's forward pipeline writes EPSG:3857 metres into local_x/local_y). A pure
    translation is fitted per candidate; the first with RMSE below FIT_ACCEPT_RMSE_M wins.
    """
    if len(samples) < 2:
        return None
    arr = np.asarray(samples)
    mean_lon = float(arr[:, 1].mean())
    south = bool(arr[:, 0].mean() < 0)
    zone = _utm_zone(mean_lon)

    candidates: list[tuple[str, str]] = []
    for z in (zone, zone - 1, zone + 1):
        if 1 <= z <= 60:
            candidates.append((f"fit-utm{z}{'S' if south else 'N'}", _utm_tmerc(z, south)))
    candidates.append(("fit-webmercator", _WEB_MERCATOR_PROJ4))

    best: Optional[tuple[float, str, str, float, float]] = None
    for name, base in candidates:
        tr = _to_latlon_transformer(base)
        x, y = tr.transform(arr[:, 0], arr[:, 1])
        dx = float(np.mean(arr[:, 2] - np.asarray(x)))
        dy = float(np.mean(arr[:, 3] - np.asarray(y)))
        res = np.hypot(np.asarray(x) + dx - arr[:, 2], np.asarray(y) + dy - arr[:, 3])
        rmse = float(np.sqrt(np.mean(res ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, name, base, dx, dy)

    if best is None or best[0] > FIT_ACCEPT_RMSE_M:
        return None
    rmse, name, base, dx, dy = best
    x0 = (_proj_param(base, "x_0") or 0.0) + dx
    y0 = (_proj_param(base, "y_0") or 0.0) + dy
    if "+proj=tmerc" in base:
        proj = re.sub(r"\+x_0=[^\s]+", f"+x_0={x0:.6f}", base)
        proj = re.sub(r"\+y_0=[^\s]+", f"+y_0={y0:.6f}", proj)
    else:
        proj = f"{base} +x_0={x0:.6f} +y_0={y0:.6f}"
    return ProjInfo(
        proj_string=proj,
        native_proj_string=proj,
        source=name,
        fit_rmse_m=rmse,
        origin_lat=_proj_param(proj, "lat_0") or 0.0,
        origin_lon=_proj_param(proj, "lon_0") or 0.0,
        scale_factor=_proj_param(proj, "k"),
    )


def proj_fallback_utm(latlon: np.ndarray) -> ProjInfo:
    lat_c, lon_c = (float(latlon[:, 0].mean()), float(latlon[:, 1].mean())) if len(latlon) else (0.0, 0.0)
    zone = _utm_zone(lon_c)
    proj = _utm_tmerc(zone, lat_c < 0)
    return ProjInfo(proj_string=proj, native_proj_string=proj, source="fallback-utm",
                    origin_lat=0.0, origin_lon=float(_utm_central_meridian(zone)),
                    scale_factor=DEFAULT_SCALE_FACTOR,
                    notes="no projector info and no usable local_x/local_y; plain UTM")


def local_frame_proj(lat_c: float, lon_c: float) -> str:
    """Map-centred transverse Mercator: coordinates near 0, k=1, exact geoReference."""
    return _tmerc(lat_c, lon_c, 1.0, 0.0, 0.0)


# --- public entry points -----------------------------------------------------------------

def resolve_projection(
    osm_root: etree._Element,
    projector_info_path: Optional[str | Path] = None,
    cli_proj: Optional[str] = None,
    frame: str = FRAME_LOCAL,
) -> ProjInfo:
    """Pick, verify and (for the local frame) re-centre the projection for one map."""
    samples = node_samples(osm_root)
    latlon = node_latlon(osm_root)

    info: Optional[ProjInfo] = None
    if cli_proj:
        info = ProjInfo(proj_string=cli_proj, native_proj_string=cli_proj, source="cli",
                        origin_lat=_proj_param(cli_proj, "lat_0"),
                        origin_lon=_proj_param(cli_proj, "lon_0"),
                        scale_factor=_proj_param(cli_proj, "k"))
    if info is None and projector_info_path and Path(projector_info_path).exists():
        yml = read_projector_info(projector_info_path)
        info = proj_from_projector_info(yml)
        if info is None:
            print(f"[l2-proj] {projector_info_path}: projector_type is local; "
                  "falling back to node fit")
    if info is None and samples:
        info = proj_from_node_fit(samples)
        if info is not None:
            print(f"[l2-proj] projection fitted from local_x/local_y: {info.source} "
                  f"(rmse {info.fit_rmse_m:.4f} m)")
    if info is None:
        info = proj_fallback_utm(latlon)
        print(f"[l2-proj] WARNING: {info.notes}; using {info.proj_string}")

    # Verify the native string against the metric node tags.
    if samples and info.native_proj_string:
        max_err, n = verify_proj(info.native_proj_string, samples)
        info.max_node_error_m, info.nodes_checked = max_err, n
        if max_err > NODE_CHECK_WARN_M:
            print(f"[l2-proj] WARNING: projection reproduces local_x/local_y only to "
                  f"{max_err:.3f} m over {n} nodes (source {info.source}); the .xodr will not "
                  "overlap the .osm exactly")

    if frame == FRAME_LOCAL:
        if len(latlon) == 0:
            raise ValueError("local frame requested but the map has no lat/lon nodes")
        lat_c = float((latlon[:, 0].min() + latlon[:, 0].max()) / 2)
        lon_c = float((latlon[:, 1].min() + latlon[:, 1].max()) / 2)
        # round so the origin is reproducible and readable
        lat_c, lon_c = round(lat_c, 6), round(lon_c, 6)
        info.frame = FRAME_LOCAL
        info.proj_string = local_frame_proj(lat_c, lon_c)
        info.origin_lat, info.origin_lon, info.scale_factor = lat_c, lon_c, 1.0
    elif frame == FRAME_MGRS:
        info.frame = FRAME_MGRS
    else:
        raise ValueError(f"unknown frame {frame!r}; expected {FRAME_MGRS!r} or {FRAME_LOCAL!r}")
    return info


def restore_latlon(osm_root: etree._Element, native_proj_string: str) -> int:
    """Fill empty ``lat``/``lon`` attributes from ``local_x``/``local_y``.

    This repo's forward pipeline (convert.py) writes ``lat="" lon=""`` after downsampling;
    crdesigner's parser needs real coordinates. Returns the number of nodes repaired.
    """
    tr = Transformer.from_crs(CRS(native_proj_string), CRS("EPSG:4326"), always_xy=True)
    fixed = 0
    for n in osm_root.findall("node"):
        if n.get("lat") and n.get("lon"):
            continue
        lx = n.find("tag[@k='local_x']")
        ly = n.find("tag[@k='local_y']")
        if lx is None or ly is None:
            raise ValueError(f"node {n.get('id')} has neither lat/lon nor local_x/local_y")
        lon, lat = tr.transform(float(lx.get("v")), float(ly.get("v")))
        n.set("lat", f"{lat:.11f}")
        n.set("lon", f"{lon:.11f}")
        fixed += 1
    return fixed


def needs_latlon_restore(osm_root: etree._Element) -> bool:
    return any(not (n.get("lat") and n.get("lon")) for n in osm_root.findall("node"))


def write_proj_info_yaml(path: str | Path, info: ProjInfo) -> None:
    payload = {k: v for k, v in asdict(info).items() if v is not None}
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
