# napari-maskel
##### Simon Wittmann, Dominik Pysch, Anna Möller
[![PyPI version](https://img.shields.io/pypi/v/napari-maskel.svg)](https://pypi.org/project/napari-maskel/)
[![Python version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

napari plugin for [maskel](https://github.com/bionetslab/maskel): skeletonization and graph-based feature extraction for branching biological structures — vasculature, fibers, neurites, and other network-like objects — with interactive visualization of branches and node features.

## Quick start

1. Install napari itself first if you haven't, with a Qt backend (e.g. `pip install "napari[all]"`), then the plugin: `pip install napari-maskel` (this also installs `maskel`).
2. In napari, open a segmentation mask (2D or 3D) and convert it to a **labels layer**.
3. Run **Analyze mask (Maskel)** from the Plugins menu, pick that layer as the input, and configure the extraction/cleanup parameters.
4. Click **Analyze mask** to add the resulting branches/nodes/summary/radius layers to the viewer, and inspect them with napari's built-in feature table widget.

![3D binary mask analyzed in napari-maskel](https://raw.githubusercontent.com/bionetslab/napari-maskel/main/docs/assets/screenshots/3d-binary-features.png)

See the [full documentation](https://bionetslab.github.io/napari-maskel/) for every configurable parameter, the 2D multi-label workflow, and sharing recipes with the [maskel](https://bionetslab.github.io/maskel/) CLI for batch processing.

## License

napari-maskel is released under the **MIT License**. See [LICENSE](LICENSE) for details.
