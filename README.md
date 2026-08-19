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

Open a binary vessel segmentation (2D or 3D) as an image layer, then run **Analyze Vessels** from the Maskel plugin menu.

Inside the widget, tune extraction settings and use **Save Config** to export a reusable JSON preset — the same preset the [maskel CLI](https://github.com/bionetslab/maskel) consumes for batch processing (`maskel run --config`).

## Tests

```sh
uv sync --extra dev && pytest
```

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
