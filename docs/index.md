napari plugin for **[maskel](https://bionetslab.github.io/maskel/)**: skeletonization and graph-based feature extraction for branching biological structures — vasculature, fibers, neurites, and other network-like objects — with interactive visualization of branches and node features.

This repo owns the interactive widget and napari-layer visualization only — the algorithm itself (thinning, feature extraction, the batch CLI) lives in maskel, which napari-maskel depends on as a normal package.

See [Installation](installation.md) to get set up, and [Widget Usage](widget-usage.md) for a tour of the **Analyze mask** widget.

## License

napari-maskel is released under the **MIT License**. See [LICENSE](https://github.com/bionetslab/napari-maskel/blob/main/LICENSE) for details.
