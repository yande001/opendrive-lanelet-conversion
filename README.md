# Map conversion - OpenDRIVE => Lanelet2

- Autoware uses Lanelet2 format for its ODD testing.
- We want more scenarios.
- ASAM OpenDRIVE dataset is a good option.

So, we need a way to convert OpenDRIVE to Lanelet2. We tried [CommonRoad Scenario Designer conversion](https://commonroad-scenario-designer.readthedocs.io/en/latest/details/open_drive/), the conversion generally worked, but the maps were containing way too many nodes, making it unusable on Autoware.

I'm attempting to fix that with custom downsampling plus Autoware-specific tagging
and topology fixes, layered on a pinned fork of crdesigner.

## Layout

- `convert.py` — single-file OpenDRIVE → Lanelet2 CLI
- `convert_l2.py` — single-file Lanelet2 → OpenDRIVE CLI (reverse direction, CARLA); `validate_xodr.py`, `roundtrip_xodr.py` — its checks
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
`--plan-view`, `--no-piecewise-width`, `--no-resample`, `--no-merge-junctions`,
`--no-sanitize-adjacency`, `--no-strip-z`, `--no-validate`, `--no-roundtrip`, `--keep-cr`.
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
