# napari-maskel

[![PyPI version](https://img.shields.io/pypi/v/napari-maskel.svg)](https://pypi.org/project/napari-maskel/)
[![Python version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

napari plugin for [maskel](https://github.com/bionetslab/maskel): mask skeletonization and graph-based feature extraction, with interactive visualization of branches and node features.

## Installation

```sh
uv sync --extra dev   # + test tools
napari
```

Before `maskel` has its first PyPI release, point uv at a local checkout instead: `uv sync --extra dev && uv pip install -e ../maskel` (adjust the path), or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.

## Usage

Open a segmentation mask (2D or 3D) as a **labels layer**, then run **Analyze Mask** from the Maskel plugin menu. A plain binary mask is treated as a single object; a multi-object instance segmentation map (more than one distinct nonzero label) is skeletonized independently per object — including two objects that touch, which stay correctly separate rather than merging into one skeleton. Every branch, node, and summary point is tagged with the `object_id` it came from, and the branches layer (the primary visualization — the raw skeleton itself isn't displayed separately) can be colored by `object_id` or any other branch property (**Branch color by** dropdown).

Inside the widget, tune extraction settings and use **Save Config** to export a reusable JSON preset — the same preset the [maskel CLI](https://github.com/bionetslab/maskel) consumes for batch processing (`maskel run --config`).

The **Spacing** field lets you set the physical pixel/voxel size (comma-separated, one value per axis) so length/radius/area features come out in physical units instead of pixel units. It's pre-filled from the selected layer's own `.scale` when you pick an image (napari has no convenient numeric field of its own to edit `.scale`, only an imprecise drag-based Transform tool or the console — this field is the convenient alternative), and you can edit it directly. An invalid value (wrong number of axes for the image, or unparsable text) shows an inline warning and falls back to pixel units for that analysis rather than failing.

## Tests

```sh
uv sync --extra dev && pytest
```

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
