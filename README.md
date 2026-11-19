# Map conversion - OpenDRIVE => Lanelet2

- Autoware uses Lanelet2 format for its ODD testing.
- We want more scenarios.
- ASAM OpenDRIVE dataset is a good option.

So, we need a way to convert OpenDRIVE to Lanelet2. We tried [CommonRoad Scenario Designer conversion](https://commonroad-scenario-designer.readthedocs.io/en/latest/details/open_drive/), the conversion generally worked, but the maps were containing way too many nodes, making it unusable on Autoware.

I'm attempting to fix that with custom downsampling plus Autoware-specific tagging
and topology fixes, layered on a pinned fork of crdesigner.

## Layout

- `convert.py` — single-file OpenDRIVE → Lanelet2 CLI
- `convert_l2.py` — single-file Lanelet2 → OpenDRIVE CLI (reverse direction, CARLA); `validate_xodr.py`, `roundtrip_xodr.py`, `carla_import_check.py` — its checks; `carla_demo.py` — recorded CARLA demo
- `demo_evan.py` — batch conversion over `sample_data/`
- `utils/` — geometry, postprocessing, config, and the `validate.py` requirement checker
- `sample_data/` — input `.xodr` maps; `output/` — generated `.osm` (gitignored)
- `extern/commonroad-scenario-designer/` — **git submodule**: the patched crdesigner
  (`autoware_dev` fork), pinned to an exact commit so results are reproducible

## Reproduce

Requires Python 3.10 and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Clone WITH the crdesigner submodule
git clone --recurse-submodules https://github.com/yande001/opendrive-lanelet-conversion.git
cd opendrive-lanelet-conversion
#    (already cloned without it? run:)
#    git submodule update --init --recursive

# 2. Create the venv and install the *pinned* dependencies
uv venv olconversion --python 3.10
uv pip install -r requirements.lock --python olconversion/bin/python

# 3. Convert a map
uv run --python olconversion/bin/python convert.py sample_data/custom/two_to_three_lanes_rht.xodr output/example.osm

# 4. (optional) Grade the output against the Autoware LL2 requirements
uv run --python olconversion/bin/python -c \
  "from utils.validate import validate_file; [print(r.line()) for r in validate_file('output/example.osm')]"
```

`requirements.lock` pins every package (including crdesigner via the submodule), so
a fresh install resolves identical versions. Do **not** `pip install
commonroad-scenario-designer` from PyPI — that release lacks the autoware_dev changes.

## OpenDRIVE → Lanelet2 ID mapping

`convert.py` writes `id_mapping_<stem>.csv` next to the output map:

```
opendrive_road_id,opendrive_lane_id,opendrive_s_min,opendrive_s_max,opendrive_direction,lanelet2_relation_id
2062,-1,0.0000,85.6534,1,1019458
```

`s_min <= s <= s_max` is the lanelet's interval on the road reference line (a scenario
position is road `s`, independent of lane travel direction); `direction` is `1` when the
lanelet runs with `+s` and `-1` against it (from the lane sign, right-hand traffic).
To get the lanelet-local `s` use the lanelet's own `centerline().length()`:

```
frac = (s - s_min) / (s_max - s_min)
if direction == -1: frac = 1 - frac
lanelet_local_s = frac * lanelet_centerline_length
```

Section bounds are exact (from the parser); pieces of a split lane section are ordered by
shared node identity and subdivided by 2D arc-length fraction, which is approximate inside a
section on bends. Roads with `singleSide` sections and groups whose pieces do not form one
chain are skipped and counted in the `ID mapping:` line. With `--concat` one lanelet spans
several sections but its tag names only the first, so rows are chain-broken or divergent.

## Lanelet2 → OpenDRIVE (reverse direction)

`convert_l2.py` converts an Autoware Lanelet2 map to OpenDRIVE 1.6 for CARLA, using the
same pinned crdesigner fork (its `lanelet2cr` / `cr2odr` paths carry fork patches, so the
PyPI release will not work here either).

```bash
# input dir with lanelet2_map.osm + map_projector_info.yaml (Autoware layout)
uv run --python olconversion/bin/python convert_l2.py ~/Downloads/r01/lanelet2_map.osm
#   -> output/r01.xodr + id_mapping_r01.csv, proj_info_r01.yaml, map_origin_r01.yaml,
#      roundtrip_r01.{png,json}

# structural checks / round-trip fidelity on their own
uv run --python olconversion/bin/python validate_xodr.py output/r01.xodr
uv run --python olconversion/bin/python roundtrip_xodr.py ~/Downloads/r01/lanelet2_map.osm output/r01.xodr --proj-info output/proj_info_r01.yaml
```

Options: `--frame local` (default: map-centred coordinates near the origin, geoReference
with `lat_0/lon_0` for CARLA GNSS) or `--frame mgrs` (x/y identical to Autoware
`local_x/local_y`); `--proj` / `--projector-info` to override the projection;
`--plan-view`, `--no-piecewise-width`, `--no-resample`, `--no-rebuild-junctions`, `--stub-len`, `--no-merge-junctions`,
`--no-sanitize-adjacency`, `--no-strip-z`, `--no-validate`, `--no-roundtrip`, `--keep-cr`.
Junctions are rebuilt from the lanelet graph for CARLA (its reader ignores `laneLink`s and
its Traffic Manager crashes on cr2odr's junction layout); `carla_import_check.py <map.xodr>`
verifies the import against a running CARLA server (needs the `carla` client wheel).
Design notes: `docs/arch/l2_to_odr.md`; full procedure (setup, stages, acceptance, CARLA
import, fork workflow, troubleshooting): `docs/l2_to_odr_procedure.md`.

## Working on crdesigner

`extern/commonroad-scenario-designer` is a normal checkout on branch `autoware_dev`.
To change converter behaviour:

```bash
cd extern/commonroad-scenario-designer
# edit, then:
git commit -am "..." && git push          # push to the crdesigner repo
cd ../..
git add extern/commonroad-scenario-designer  # bump the pinned pointer
git commit -m "Bump crdesigner submodule"
```

It is installed editable, so edits take effect immediately without reinstalling.
