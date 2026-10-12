"""
Compare a generated OpenDRIVE file with the Lanelet2 map it was converted from.
See utils/roundtrip.py for the metrics.

Usage:
    python roundtrip_xodr.py <orig.osm> <out.xodr> [--proj STRING | --proj-info proj_info_<stem>.yaml]
                             [--png overlay.png] [--json metrics.json]
If neither --proj nor --proj-info is given, the .xodr <geoReference> is used.
"""
import argparse
import sys

import yaml
from lxml import etree

from utils.roundtrip import roundtrip_compare


def main():
    parser = argparse.ArgumentParser(description="Round-trip check: .osm vs generated .xodr")
    parser.add_argument("osm", help="Original Lanelet2 .osm")
    parser.add_argument("xodr", help="Generated OpenDRIVE .xodr")
    parser.add_argument("--proj", help="PROJ string the .xodr coordinates are in")
    parser.add_argument("--proj-info", help="proj_info_<stem>.yaml written by convert_l2.py")
    parser.add_argument("--png", help="Overlay plot path")
    parser.add_argument("--json", help="Write metrics as JSON to this path")
    args = parser.parse_args()

    proj = args.proj
    if proj is None and args.proj_info:
        with open(args.proj_info) as f:
            proj = yaml.safe_load(f)["proj_string"]
    if proj is None:
        geo = etree.parse(args.xodr).getroot().find("header/geoReference")
        proj = (geo.text or "").strip() if geo is not None else ""
    if not proj:
        print("Error: no projection given and the .xodr has no <geoReference>")
        sys.exit(1)

    metrics = roundtrip_compare(args.osm, args.xodr, proj, png_path=args.png)
    for k, v in metrics.summary().items():
        print(f"  {k:<28} {v}")
    print(f"  worst matches (osm id, IoU):  {metrics.worst}")
    if args.json:
        metrics.to_json(args.json)
        print(f"  metrics written to {args.json}")


if __name__ == "__main__":
    main()
