"""High-level napari-layer extraction API."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from maskel.config import ExtractionConfig
from maskel.pipeline import AnalysisResult, ObjectResult

if TYPE_CHECKING:
    from napari.types import LayerDataTuple  # noqa: F401


def extract_skeleton_layers(
    result: AnalysisResult,
    base_name: str,
    config: ExtractionConfig | None = None,
) -> list["napari.types.LayerDataTuple"]:  # noqa: F821
    """Build napari visualization layers from a maskel `AnalysisResult`.

    Parameters
    ----------
    result : AnalysisResult
        Output of `maskel.pipeline.analyze_segmentation_mask`. May cover more
        than one object (see `AnalysisResult`/`ObjectResult`); layers combine
        all objects, tagged with ``object_id`` as a layer property.
    base_name : str
        Base name for layer naming.
    config : ExtractionConfig, optional
        Configuration for what to extract. Defaults to all except fractal_dimension.
    """
    if config is None:
        config = ExtractionConfig()

    layers = []

    if config.branches:
        branch_layer = _extract_branch_features_layer(
            base_name,
            result.objects,
            color_property=config.branch_color_property,
        )
        if branch_layer is not None:
            layers.append(branch_layer)

            if config.branch_text:
                text_layer = _extract_branch_text_layer(branch_layer, base_name)
                layers.append(text_layer)

    if config.nodes:
        node_layer = _extract_node_features_layer(base_name, result.node_records)
        if node_layer is not None:
            layers.append(node_layer)

    if config.summary:
        summary_layer = _extract_summary_features_layer(base_name, result.objects)
        if summary_layer is not None:
            layers.append(summary_layer)

    if result.radius_matrix is not None:
        radius_layer = _extract_radius_layer(
            result.radius_matrix, result.skeleton, base_name
        )
        if radius_layer is not None:
            layers.append(radius_layer)

    return layers


def _extract_radius_layer(
    radius_matrix: np.ndarray,
    skeleton: np.ndarray,
    base_name: str,
) -> "napari.types.LayerDataTuple | None":  # noqa: F821
    """Create an image layer showing per-pixel mask radius on the skeleton."""
    if not np.any(radius_matrix):
        return None

    display = np.where(skeleton > 0, radius_matrix, np.nan)
    meta = {
        "name": f"{base_name}_radius",
        "colormap": "turbo",
        "blending": "additive",
        "opacity": 0.85,
    }
    return (display, meta, "image")


def _extract_branch_features_layer(
    base_name: str,
    objects: list[ObjectResult],
    color_property: str = "tortuosity",
) -> "napari.types.LayerDataTuple | None":  # noqa: F821
    """Build one combined branch-paths layer spanning all objects.

    Parameters
    ----------
    base_name : str
        Base name used for layer naming.
    objects : list[ObjectResult]
        Per-object results (from `AnalysisResult.objects`). Branch path
        coordinates are offset into global image coordinates using each
        object's own crop offset.
    color_property : str
        Branch property to use for edge coloring (including ``object_id``).
        Must be a numeric column. Defaults to "tortuosity".

    Returns
    -------
    LayerDataTuple or None
        Napari shapes layer for branch paths, or None if no object has any
        branches.
    """
    frames = []
    path_data = []

    for obj in objects:
        if obj.graph is None or obj.branch_data is None or obj.branch_data.empty:
            continue
        branch_data = obj.branch_data.reset_index(drop=True).copy()
        branch_data["branch_id"] = np.arange(len(branch_data), dtype=np.int64)
        branch_data["object_id"] = obj.object_id

        offset = np.array(obj.offset, dtype=float)
        for i in range(len(branch_data)):
            path_data.append(obj.graph.path_coordinates(i) + offset)

        frames.append(branch_data)

    if not frames:
        return None

    branch_data = pd.concat(frames, ignore_index=True)

    meta = {
        "name": f"{base_name}_branches",
        "shape_type": "path",
        "properties": branch_data,
        "face_color": "transparent",
        "edge_width": 0.5,
        "opacity": 0.95,
    }

    values = branch_data.get(color_property)
    if values is not None:
        numeric = np.asarray(values, dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size > 1 and float(np.min(finite)) < float(np.max(finite)):
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            meta["edge_color"] = color_property
            meta["edge_colormap"] = "turbo"
            meta["edge_contrast_limits"] = (vmin, vmax)
            return (path_data, meta, "shapes")

    meta["edge_color"] = "#30d5c8"
    return (path_data, meta, "shapes")


def _extract_branch_text_layer(
    branch_layer: "napari.types.LayerDataTuple",  # noqa: F821
    base_name: str,
) -> "napari.types.LayerDataTuple":  # noqa: F821
    path_data = branch_layer[0]
    branch_data = branch_layer[1]["properties"]

    label_points = []
    for coords in path_data:
        if len(coords) == 0:
            label_points.append(np.zeros((coords.shape[1],), dtype=float))
            continue
        label_points.append(np.asarray(coords, dtype=float).mean(axis=0))

    points = np.asarray(label_points, dtype=float)
    meta = {
        "name": f"{base_name}_branch_text",
        "properties": branch_data,
        "symbol": "disc",
        "size": 1,
        "face_color": "transparent",
        "border_color": "transparent",
        "opacity": 1.0,
        "text": {
            "string": "obj {object_id} | id {branch_id} | L={branch-distance:.1f} | T={tortuosity:.2f}",
            "size": 9,
            "color": "white",
            "anchor": "center",
        },
    }
    return (points, meta, "points")


def _extract_summary_features_layer(
    base_name: str,
    objects: list[ObjectResult],
) -> "napari.types.LayerDataTuple | None":  # noqa: F821
    """Create a summary point layer, one point per object, at that object's
    own skeleton-graph centroid (in global image coordinates).

    Parameters
    ----------
    base_name : str
        Base name for layer naming.
    objects : list[ObjectResult]
        Per-object results (from `AnalysisResult.objects`). An object with no
        summary features (summary disabled) or no graph (fully empty
        skeleton) contributes no point.
    """
    points = []
    rows = []
    for obj in objects:
        if not obj.summary_features or obj.graph is None:
            continue
        center = obj.graph.coordinates.mean(axis=0) + np.array(obj.offset, dtype=float)
        points.append(center)
        rows.append(obj.summary_features)

    if not points:
        return None

    meta_features = {k: [r.get(k) for r in rows] for k in rows[0]}
    meta = {
        "name": f"{base_name}_summary",
        "properties": meta_features,
        "symbol": "ring",
        "size": 8,
        "face_color": "transparent",
        "border_color": "yellow",
        "opacity": 0.9,
        "text": {
            "string": "object {object_id}",
            "size": 10,
            "color": "yellow",
            "anchor": "upper_left",
        },
    }
    return (np.asarray(points, dtype=float), meta, "points")


def _extract_node_features_layer(
    base_name: str,
    node_records: list[dict[str, object]],
) -> "napari.types.LayerDataTuple | None":  # noqa: F821
    """Create a points layer showing graph nodes colored by degree.

    Parameters
    ----------
    base_name : str
        Base name for layer naming.
    node_records : list[dict[str, object]]
        Pre-computed node records (e.g. from `AnalysisResult.node_records`),
        already in global image coordinates and tagged with ``object_id``.
    """
    if not node_records:
        return None

    ndim = sum(1 for k in node_records[0] if k.startswith("coord_"))
    points = np.array(
        [tuple(r[f"coord_{d}"] for d in range(ndim)) for r in node_records], dtype=float
    )

    props = {
        k: [r[k] for r in node_records]
        for k in node_records[0]
        if not k.startswith("coord_")
    }

    meta = {
        "name": f"{base_name}_nodes",
        "properties": props,
        "symbol": "disc",
        "size": 3,
        "face_color": "degree",
        "face_colormap": "viridis",
        "opacity": 0.8,
    }
    return (points, meta, "points")
