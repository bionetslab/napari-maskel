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

To test against an unreleased `maskel` checkout instead, point uv at it locally: `uv sync --extra dev && uv pip install -e ../maskel` (adjust the path), or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.

## Usage

Open a segmentation mask (2D or 3D) as a **labels layer**, then run **Analyze mask** from the Maskel plugin menu. A plain binary mask is treated as a single object; a multi-object instance segmentation map (more than one distinct nonzero label) is skeletonized independently per object — including two objects that touch, which stay correctly separate rather than merging into one skeleton. Every branch, node, and summary point is tagged with the `object_id` it came from, and the branches layer (the primary visualization — the raw skeleton itself isn't displayed separately) can be colored by `object_id` or any other branch property (**Branch color by** dropdown).

Inside the widget, tune extraction settings and use **Save recipe** to export a reusable JSON preset — the same preset the [maskel CLI](https://github.com/bionetslab/maskel) consumes for batch processing (`maskel run --config`). The widget scrolls, so it's never clipped by the dock panel's height; every group (including the less-frequently-used **Cleanup** and **Advanced features**) is its own collapsible section, expanded by default (click a header to collapse it).

The **Spacing** field lets you set the physical pixel/voxel size (comma-separated, one value per axis) so length/radius/area features come out in physical units instead of pixel units. It's pre-filled from the selected layer's own `.scale` when you pick an image (napari has no convenient numeric field of its own to edit `.scale`, only an imprecise drag-based Transform tool or the console — this field is the convenient alternative), and you can edit it directly. An invalid value (wrong number of axes for the image, or unparsable text) shows an inline warning and falls back to pixel units for that analysis rather than failing. Once an image is selected, its shape is shown right above the field so it's clear which axis order to enter spacing in.

The node layer (when **Extract node features** is on) only shows branch/end nodes (degree ≠ 2) — pass-through points along a straight, unbranched run are omitted from the visualization so it reads as a graph alongside the branches layer, though the exported node CSV still includes every node.

**Write networkx graph (.pkl)** exports the same skeleton graph as **Write graph (.graphml)**, but as a pickled `networkx.MultiGraph` rather than GraphML/XML — richer, since it keeps NaN attributes and native Python types that GraphML can't represent, at the cost of only being readable from Python (`pickle.load`).

## Tests

```sh
uv sync --extra dev && pytest
```

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
