# napari-maskel

[![PyPI version](https://img.shields.io/pypi/v/napari-maskel.svg)](https://pypi.org/project/napari-maskel/)
[![Python version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

napari plugin for [maskel](https://github.com/bionetslab/maskel): skeletonization and graph-based feature extraction for branching biological structures — vasculature, fibers, neurites, and other network-like objects — with interactive visualization of branches and node features.

**Full documentation: https://bionetslab.github.io/napari-maskel/**

## Installation

```sh
uv sync --extra dev   # + test tools
napari
```

To test against an unreleased `maskel` checkout instead, point uv at it locally: `uv sync --extra dev && uv pip install -e ../maskel` (adjust the path), or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.

## Usage

Open a segmentation mask (2D or 3D) as a **labels layer**, then run **Analyze mask** from the Maskel plugin menu. See the [Widget Usage](https://bionetslab.github.io/napari-maskel/widget-usage/) docs for a full tour, including config sharing with the CLI, physical spacing, and graph export.

## Tests

```sh
uv sync --extra dev && pytest
```

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
