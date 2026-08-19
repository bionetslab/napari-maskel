# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

napari-maskel: the napari plugin for [maskel](https://github.com/bionetslab/maskel) (core skeletonization/feature-extraction package, installed as a normal dependency). This repo owns the interactive widget and the napari-layer visualization — nothing algorithmic lives here.

## Commands

```sh
uv sync --extra dev
uv run pytest
uvx ruff check
uvx ruff format --check
uv build
```

## Architecture

**Widget calls the core pipeline, then builds its own layers.** [napari_maskel/_napari.py](napari_maskel/_napari.py)'s `VesselAnalysisWidget._on_analyze` calls `maskel.pipeline.analyze_binary_image()` to get a plain-data `AnalysisResult`, then passes its fields (`skeleton`, `graph`, `branch_data`, `summary_features`, `radius_matrix`) into [napari_maskel/napari_layers.py](napari_maskel/napari_layers.py)'s `extract_skeleton_layers()` to build napari `LayerDataTuple`s. `AnalysisResult` itself has no `layers` field — that boundary is deliberate, keeping `maskel` napari-free. Only build layers when `result.graph is not None` (an empty skeleton short-circuits `analyze_binary_image` before a graph exists).

**Config is shared with the CLI.** `PipelineConfig`/`ExtractionConfig`/`OutputConfig` come from `maskel.config` and are the same schema the `maskel` CLI reads/writes. The widget's **Save Config**/**Load Config** buttons round-trip the exact JSON `maskel run --config` consumes — don't add plugin-only config fields here; add them to `maskel.config` instead so both stay in sync.

**napari.yaml is the manifest.** [napari_maskel/napari.yaml](napari_maskel/napari.yaml)'s `version` must match `pyproject.toml`'s `[project].version` — bump both together on release.

## Tests

`tests/test_napari_layers.py` tests `extract_skeleton_layers()` directly with plain arrays/graphs — it doesn't need napari itself running, only `maskel`'s graph-building helpers.
