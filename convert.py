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

# --- Constants ---
PROJ_DEG = "EPSG:4326"
PROJ_MET = "EPSG:3857"
R = 6378000  # Earth radius in meters

# Downsampling defaults
DEFAULT_ANGLE_THRSH = 179.9
DEFAULT_MIN_DIST = 3.0

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
_FEATURE_WAY_TYPES = {"light_bulbs"}
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


def write_id_mapping(converter, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opendrive_road_id", "opendrive_section_id", "opendrive_lane_id", "lanelet2_relation_id"])
        for (road_id, section_id, lane_id), l2_id in sorted(converter.odr_to_l2_mapping.items()):
            w.writerow([road_id, section_id, lane_id, l2_id])


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

    # ID mapping
    mapping_path = output_path.with_name(f"id_mapping_{stem}.csv")
    write_id_mapping(converter, mapping_path)
    print(f"ID mapping:     {mapping_path} ({len(converter.odr_to_l2_mapping)} entries)")

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

    # Tag unmarked physical road edges as road_border (vm-01-02). Runs last so
    # every final boundary-way piece (post-split) is classified. Interior unmarked
    # dividers between two road lanes are left untyped by policy.
    if not args.no_road_border:
        bstats = add_road_borders(osm)
        print(f"Road borders:   tagged {bstats['borders_tagged']} unmarked edges "
              f"as road_border")

    write_osm(osm, output_path)
    print(f"Output:         {output_path}")


if __name__ == "__main__":
    main()
