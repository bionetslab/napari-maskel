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

**Widget calls the core pipeline, then builds its own layers.** [napari_maskel/_napari.py](napari_maskel/_napari.py)'s `VesselAnalysisWidget._on_analyze` calls `maskel.pipeline.analyze_binary_image()` to get a plain-data `AnalysisResult`, then passes the whole result into [napari_maskel/napari_layers.py](napari_maskel/napari_layers.py)'s `extract_skeleton_layers(result, base_name, config)` to build napari `LayerDataTuple`s. `AnalysisResult` itself has no `layers` field — that boundary is deliberate, keeping `maskel` napari-free. `extract_skeleton_layers` is safe to call unconditionally (an empty result just produces an empty layer list).

**Input is a Labels layer, not an Image layer**, since `maskel` now accepts multi-object instance segmentation maps (each distinct nonzero value its own object) as well as plain binary masks — the `_extraction_params` magicgui signature in `_napari.py` types `image` as `napari.layers.Labels`.

**One `AnalysisResult` can cover many objects.** `result.object_graphs` is a list of `ObjectGraph` (one per object: its own `skan.Skeleton`, local branch-data DataFrame, and crop offset), not a single graph — `napari_layers.py`'s branch/node/summary builders loop over it, offsetting each object's local coordinates into global image space, and concatenate into one combined layer per feature type (not one layer per object) so `object_id` is just another colorable/filterable property. Don't reach into a single `graph`/`branch_data` field on `AnalysisResult` — those don't exist; `branch_records`/`node_records` (flat, already global-coordinate, already `object_id`-tagged) or `object_graphs` are the way in.

**Config is shared with the CLI.** `PipelineConfig`/`ExtractionConfig`/`OutputConfig` come from `maskel.config` and are the same schema the `maskel` CLI reads/writes. The widget's **Save Config**/**Load Config** buttons round-trip the exact JSON `maskel run --config` consumes — don't add plugin-only config fields here; add them to `maskel.config` instead so both stay in sync.

**napari.yaml is the manifest.** [napari_maskel/napari.yaml](napari_maskel/napari.yaml)'s `version` must match `pyproject.toml`'s `[project].version` — bump both together on release.

## Tests

`tests/test_napari_layers.py` builds real `AnalysisResult`s via `maskel.pipeline.analyze_binary_image()` on small synthetic masks (including multi-object ones) and feeds them straight into `extract_skeleton_layers()` — it doesn't need napari itself running, just `maskel` installed.
