"""
Validate a generated OpenDRIVE `.xodr` (structure, links, geometry continuity).
See utils/validate_xodr.py for the checks.

Usage:
    python validate_xodr.py <map.xodr> [<map2.xodr> ...] [--json]
"""
import argparse
import json
import sys

from utils.validate_xodr import validate_file
from utils.validate import PASS, FAIL, SKIP


def main():
    parser = argparse.ArgumentParser(description="Validate OpenDRIVE .xodr files.")
    parser.add_argument("maps", nargs="+", help="Path(s) to .xodr file(s)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    all_json = {}
    any_fail = False
    for path in args.maps:
        results = validate_file(path)
        counts = {PASS: 0, FAIL: 0, SKIP: 0}
        for r in results:
            counts[r.status] += 1
        if counts[FAIL]:
            any_fail = True
        if args.json:
            all_json[path] = [
                {"req_id": r.req_id, "name": r.name, "status": r.status,
                 "detail": r.detail, "offenders": r.offenders, "total": r.total}
                for r in results
            ]
        else:
            print(f"\nMap: {path}")
            for r in results:
                print(r.line())
            print(f"  ── {counts[PASS]} pass, {counts[FAIL]} fail, {counts[SKIP]} skip")

    if args.json:
        print(json.dumps(all_json, indent=2))
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
