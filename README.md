# napari-maskel

[![PyPI version](https://img.shields.io/pypi/v/napari-maskel.svg)](https://pypi.org/project/napari-maskel/)
[![Python version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

napari plugin for [maskel](https://github.com/bionetslab/maskel): vessel skeletonization and graph-based feature extraction, with interactive visualization of skeletons, branches, and node features.

## Installation

```sh
uv sync --extra dev   # + test tools
napari
```

Before `maskel` has its first PyPI release, point uv at a local checkout instead: `uv sync --extra dev && uv pip install -e ../maskel` (adjust the path), or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.

## Usage

Open a vessel segmentation (2D or 3D) as a **labels layer**, then run **Analyze Vessels** from the Maskel plugin menu. A plain binary mask is treated as a single object; a multi-object instance segmentation map (more than one distinct nonzero label) is skeletonized independently per object — including two objects that touch, which stay correctly separate rather than merging into one skeleton. Every branch, node, and summary point is tagged with the `object_id` it came from, and the branches layer can be colored by `object_id` (**Branch color by** dropdown).

Inside the widget, tune extraction settings and use **Save Config** to export a reusable JSON preset — the same preset the [maskel CLI](https://github.com/bionetslab/maskel) consumes for batch processing (`maskel run --config`).

## Tests

```sh
uv sync --extra dev && pytest
```

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
