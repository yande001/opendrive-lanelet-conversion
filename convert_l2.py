"""
Convert a single Autoware Lanelet2 (.osm) file to OpenDRIVE (.xodr).

Usage:
    python convert_l2.py <input.osm> [output.xodr] [options]

If output path is omitted, writes to ./output/<stem>.xodr, where <stem> is the file
stem, or the parent directory name when the file is called lanelet2_map.osm (the
Autoware convention: ~/Downloads/r01/lanelet2_map.osm -> output/r01.xodr).

Pipeline (each stage has a --no-X switch and prints one stats line):
    projection  -> crdesigner lanelet_to_commonroad -> strip z -> sanitize adjacency
    -> resample boundaries -> crdesigner cr2odr Converter -> validate -> round-trip check

Reverse counterpart of convert.py; see docs/arch/l2_to_odr.md.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

from lxml import etree

# Use the pinned crdesigner submodule (extern/commonroad-scenario-designer); editable-installed,
# the sys.path insert is a fallback for a fresh checkout (same as convert.py).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CR_DESIGNER_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "extern", "commonroad-scenario-designer"))
sys.path.insert(0, CR_DESIGNER_PATH)

from crdesigner.common.config.general_config import general_config
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.config.opendrive_config import open_drive_config
from crdesigner.map_conversion.map_conversion_interface import lanelet_to_commonroad
from crdesigner.map_conversion.opendrive.cr2odr.converter import Converter

from utils.cr_prep import resample_boundaries, sanitize_adjacency, strip_z, topology_report
from utils.l2_projection import (
    FRAME_LOCAL,
    FRAME_MGRS,
    ProjInfo,
    needs_latlon_restore,
    resolve_projection,
    restore_latlon,
    write_proj_info_yaml,
)
from utils.map_origin import write_map_origin_yaml
from utils.xodr_post import postprocess_file

DEFAULT_RESAMPLE_STEP = 1.0
PLAN_VIEW_MODES = ("param_poly3", "primitives")
AUTOWARE_MAP_FILENAME = "lanelet2_map.osm"
AUTOWARE_PROJECTOR_INFO = "map_projector_info.yaml"


# --- Stages ---

def output_stem(input_path: Path) -> str:
    if input_path.name == AUTOWARE_MAP_FILENAME:
        return input_path.resolve().parent.name
    return input_path.stem


def load_scenario(osm_path: Path, proj: ProjInfo):
    """Run crdesigner's Lanelet2 -> CommonRoad conversion under the chosen projection.

    Returns (scenario, old_ids) where old_ids maps the Lanelet2 relation id (str) to the
    CommonRoad lanelet id (int).
    """
    # The converter reads the module-global general_config (not the instance passed in).
    general_config.proj_string_cr = proj.proj_string
    lanelet2_config.translate = False
    lanelet2_config.autoware = False
    scenario = lanelet_to_commonroad(str(osm_path), general_conf=general_config, lanelet2_conf=lanelet2_config)
    if scenario is None:
        return None, {}
    old_ids = dict(getattr(scenario.lanelet_network, "_old_lanelet_ids", {}))
    return scenario, old_ids


def convert_scenario_to_xodr(scenario, output_path: Path, plan_view: str = PLAN_VIEW_MODES[0],
                             piecewise_width: bool = True):
    """CommonRoad -> OpenDRIVE via the patched cr2odr; returns the converter (id bookkeeping)."""
    # fork additions (config attributes are class-level, i.e. process-wide)
    open_drive_config.cr2odr_plan_view = plan_view
    open_drive_config.cr2odr_piecewise_width = piecewise_width
    converter = Converter(scenario)
    converter.convert(str(output_path))
    return converter


def write_id_mapping(path: Path, old_ids: dict, converter) -> int:
    """CSV: lanelet2_relation_id, commonroad_id, opendrive_road_id, opendrive_lane_id."""
    cr_to_road = getattr(converter, "cr_id_to_od", {})
    cr_to_lane = getattr(converter, "lanelet_to_lane", {})
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lanelet2_relation_id", "commonroad_id", "opendrive_road_id", "opendrive_lane_id"])
        for osm_id, cr_id in sorted(old_ids.items(), key=lambda kv: kv[1]):
            w.writerow([osm_id, cr_id, cr_to_road.get(cr_id, ""), cr_to_lane.get(cr_id, "")])
            rows += 1
    return rows


def _fmt(stats: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in stats.items())


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Convert a single Lanelet2 (.osm) file to OpenDRIVE (.xodr).")
    parser.add_argument("input", help="Path to .osm file")
    parser.add_argument("output", nargs="?", help="Output .xodr path (default: output/<stem>.xodr)")
    parser.add_argument("--projector-info", help="Autoware map_projector_info.yaml "
                        f"(default: {AUTOWARE_PROJECTOR_INFO} next to the input, if present)")
    parser.add_argument("--proj", help="PROJ string to use instead of deriving one (lat/lon -> x/y)")
    parser.add_argument("--frame", choices=[FRAME_LOCAL, FRAME_MGRS], default=FRAME_LOCAL,
                        help=f"'{FRAME_LOCAL}': map-centred tmerc, coordinates near the origin (CARLA); "
                             f"'{FRAME_MGRS}': x/y identical to Autoware local_x/local_y (default: {FRAME_LOCAL})")
    parser.add_argument("--no-strip-z", action="store_true", help="Keep 3-D vertices (cr2odr will fail on them)")
    parser.add_argument("--no-sanitize-adjacency", action="store_true", help="Skip adjacency repair")
    parser.add_argument("--no-resample", action="store_true", help="Skip boundary resampling")
    parser.add_argument("--resample-step", type=float, default=DEFAULT_RESAMPLE_STEP,
                        help=f"Target vertex spacing in metres (default {DEFAULT_RESAMPLE_STEP})")
    parser.add_argument("--plan-view", choices=PLAN_VIEW_MODES, default=PLAN_VIEW_MODES[0],
                        help="Reference-line geometry: 'param_poly3' = one exact cubic per segment "
                             "(default); 'primitives' = crdesigner's line/arc/spiral classification")
    parser.add_argument("--no-piecewise-width", action="store_true",
                        help="Single linear lane-width fit (crdesigner default) instead of one piece per sample")
    parser.add_argument("--no-merge-junctions", action="store_true",
                        help="Keep crdesigner's one-junction-per-incoming-road output")
    parser.add_argument("--no-validate", action="store_true", help="Skip the XML-level .xodr checks")
    parser.add_argument("--no-roundtrip", action="store_true",
                        help="Skip the xodr->CommonRoad round-trip comparison and overlay plot")
    parser.add_argument("--keep-cr", action="store_true", help="Also write the intermediate CommonRoad XML")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    stem = output_stem(input_path)
    output_path = Path(args.output) if args.output else Path("output") / f"{stem}.xodr"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projector_info = args.projector_info
    if projector_info is None:
        candidate = input_path.parent / AUTOWARE_PROJECTOR_INFO
        projector_info = str(candidate) if candidate.exists() else None

    # Projection
    osm_root = etree.parse(str(input_path)).getroot()
    proj = resolve_projection(osm_root, projector_info, args.proj, args.frame)
    print(f"Projection:     {proj.source} / frame={proj.frame}")
    print(f"                {proj.proj_string}")
    if proj.nodes_checked:
        print(f"                local_x/y reproduced to {proj.max_node_error_m:.4f} m over {proj.nodes_checked} nodes")

    conversion_input = input_path
    if needs_latlon_restore(osm_root):
        if not proj.native_proj_string:
            print("Error: nodes lack lat/lon and no projection for local_x/local_y could be derived")
            sys.exit(1)
        proj.restored_latlon = restore_latlon(osm_root, proj.native_proj_string)
        conversion_input = output_path.with_name(f"normalized_{stem}.osm")
        etree.ElementTree(osm_root).write(str(conversion_input), xml_declaration=True, encoding="UTF-8")
        print(f"Normalized:     {conversion_input} ({proj.restored_latlon} nodes given lat/lon from local_x/y)")

    proj_path = output_path.with_name(f"proj_info_{stem}.yaml")
    write_proj_info_yaml(proj_path, proj)
    print(f"Proj info:      {proj_path}")
    if proj.frame == FRAME_LOCAL:
        origin_path = output_path.with_name(f"map_origin_{stem}.yaml")
        write_map_origin_yaml(origin_path, proj.to_map_origin())
        print(f"Map origin:     {origin_path}")

    # Lanelet2 -> CommonRoad
    print(f"Converting {input_path} ...")
    scenario, old_ids = load_scenario(conversion_input, proj)
    if scenario is None:
        print("Conversion failed (empty scenario).")
        sys.exit(1)
    print(f"CommonRoad:     {_fmt(topology_report(scenario))}")

    # Scenario preparation
    if not args.no_strip_z:
        zstats, _z_table = strip_z(scenario)
        print(f"Strip z:        {_fmt(zstats)}")
    if not args.no_sanitize_adjacency:
        astats = sanitize_adjacency(scenario)
        print(f"Adjacency:      {_fmt(astats)}")
    if not args.no_resample:
        rstats = resample_boundaries(scenario, step=args.resample_step)
        print(f"Resample:       {_fmt(rstats)}")

    if args.keep_cr:
        from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
        from commonroad.planning.planning_problem import PlanningProblemSet
        cr_path = output_path.with_name(f"cr_{stem}.xml")
        CommonRoadFileWriter(scenario, PlanningProblemSet()).write_to_file(
            str(cr_path), OverwriteExistingFile.ALWAYS)
        print(f"CommonRoad XML: {cr_path}")

    # CommonRoad -> OpenDRIVE
    converter = convert_scenario_to_xodr(scenario, output_path, args.plan_view, not args.no_piecewise_width)
    if not args.no_merge_junctions:
        jstats = postprocess_file(output_path, merge=True)
        print(f"Junctions:      {_fmt(jstats)}")
    xodr_root = etree.parse(str(output_path)).getroot()
    print(f"OpenDRIVE:      roads={len(xodr_root.findall('road'))} "
          f"junctions={len(xodr_root.findall('junction'))} "
          f"geometries={len(xodr_root.findall('road/planView/geometry'))}")

    mapping_path = output_path.with_name(f"id_mapping_{stem}.csv")
    rows = write_id_mapping(mapping_path, old_ids, converter)
    print(f"ID mapping:     {mapping_path} ({rows} lanelets)")

    # Verification (never fatal: the .xodr is already written)
    if not args.no_validate:
        from utils.validate_xodr import validate_file
        results = validate_file(output_path)
        failed = [r for r in results if r.status == "FAIL"]
        for r in results:
            print(f"                {r.line()}")
        print(f"Validate:       {len(results) - len(failed)}/{len(results)} checks pass")
    if not args.no_roundtrip:
        from utils.roundtrip import roundtrip_compare
        png_path = output_path.with_name(f"roundtrip_{stem}.png")
        try:
            metrics = roundtrip_compare(conversion_input, output_path, proj.proj_string, png_path=png_path)
        except Exception as e:  # noqa: BLE001 - crdesigner's readers raise plain Exceptions
            print(f"Round-trip:     SKIPPED (crdesigner could not read the map back: {e})")
        else:
            metrics.to_json(output_path.with_name(f"roundtrip_{stem}.json"))
            print(f"Round-trip:     {_fmt(metrics.summary())}")
            print(f"                overlay {png_path}")

    print(f"Output:         {output_path}")


if __name__ == "__main__":
    main()
