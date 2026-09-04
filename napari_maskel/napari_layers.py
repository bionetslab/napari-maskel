"""High-level napari-layer extraction API."""

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from maskel.config import ExtractionConfig
from maskel.pipeline import AnalysisResult, ObjectResult
from napari.layers import Points

if TYPE_CHECKING:
    from napari.types import LayerDataTuple  # noqa: F401


def parse_spacing_input(
    text: str, ndim: int
) -> tuple[tuple[float, ...] | None, str | None]:
    """Parse a comma-separated spacing string, validating its length against *ndim*.

    A standalone, pure function (no Qt/napari dependency) so it's
    unit-testable without a real napari viewer - the widget calls this both
    to validate on change and to build the `ExtractionConfig` at analyze
    time, showing an inline warning label instead of the CLI's stderr
    warning when the input is invalid.

    Parameters
    ----------
    text : str
        Comma-separated spacing values, e.g. ``"1.0, 0.5, 0.5"``. Napari
        layers always have a ``.scale`` of length ``ndim`` (defaulting to
        all ``1.0``), which is the natural source for the field's initial
        value; this function just parses/validates whatever the user has
        typed there.
    ndim : int
        Expected number of axes (the selected image layer's dimensionality).

    Returns
    -------
    spacing : tuple[float, ...] or None
        The parsed spacing, or ``None`` if *text* is blank/whitespace-only
        (treated as "no override" - valid, matching the CLI/pipeline's own
        None-means-isotropic default) or invalid.
    error : str or None
        A human-readable message when *text* is invalid (unparsable, or the
        wrong number of values for *ndim*), else ``None``.
    """
    stripped = text.strip()
    if not stripped:
        return None, None

    parts = [p.strip() for p in stripped.split(",")]
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        return None, f"Spacing must be comma-separated numbers, got '{text}'"

    if len(values) != ndim:
        return None, (
            f"Spacing has {len(values)} value(s) but the image has {ndim} dimension(s)"
        )

    return values, None


# Points layers clamp their on-screen size to this range in canvas pixels
# (napari's Points `canvas_size_limits`), so branch/end nodes and per-object
# summary markers stay visible regardless of image size or zoom - without
# it, a fixed *data-space* point size (the same units as image pixels)
# renders as an imperceptible dot on a multi-thousand-pixel image like a
# full-resolution fundus photo, even though it looks fine on a small test
# image. Shapes (the branches path layer) has no such canvas-space option
# in napari, so its edge_width is instead computed proportionally to the
# image's own size - see _proportional_edge_width.
_POINT_CANVAS_SIZE_LIMITS = (4.0, 30.0)


# napari[all]'s own version floor can't be pinned to guarantee this kwarg is
# present: on Intel Mac, napari[all]'s numba upper bound conflicts with
# maskel's numba floor, so that platform is stuck on an older napari where
# Points never gained canvas_size_limits. Passing an unsupported kwarg to
# Layer.create doesn't just skip the sizing behavior - it raises, and the
# whole points layer silently fails to be added (caught by the per-layer
# except in _on_analyze). Checking the real constructor signature at import
# time means every platform still gets a points layer, just without
# canvas-space size clamping on the (increasingly rare) old-napari one.
def _points_supports_canvas_size_limits(points_cls) -> bool:
    """Whether *points_cls* (the real napari.layers.Points, or a fake for
    testing) accepts a ``canvas_size_limits`` keyword."""
    return "canvas_size_limits" in inspect.signature(points_cls.__init__).parameters


_POINTS_SUPPORTS_CANVAS_SIZE_LIMITS = _points_supports_canvas_size_limits(Points)
_POINT_SIZE_KWARGS = (
    {"canvas_size_limits": _POINT_CANVAS_SIZE_LIMITS}
    if _POINTS_SUPPORTS_CANVAS_SIZE_LIMITS
    else {}
)


def _proportional_edge_width(image_shape: tuple[int, ...]) -> float:
    """A branch-path edge_width (data-space units) that scales with the
    image's own size, floored at 1.0.

    Shapes' edge_width has no canvas-pixel-space option (unlike Points'
    canvas_size_limits), so a fixed absolute value that looks right on a
    small synthetic test image (tens of pixels) renders as a barely-visible,
    often dotted line on a full-resolution image spanning thousands of
    pixels - and the reverse: a value tuned for a large image would look
    like a thick blob on a small one. /500 keeps a ~3500px-wide fundus
    photo's branches a few screen pixels wide at a typical fit-to-window
    zoom (mean branch length there is ~140px, so this stays comfortably
    thinner than even a short branch) while still flooring to something
    visible on tiny images.
    """
    return max(1.0, max(image_shape) / 500.0)


def extract_skeleton_layers(
    result: AnalysisResult,
    base_name: str,
    config: ExtractionConfig | None = None,
) -> list["napari.types.LayerDataTuple"]:  # noqa: F821, UP037
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
            result.skeleton.shape,
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
) -> "napari.types.LayerDataTuple | None":  # noqa: F821, UP037
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
    image_shape: tuple[int, ...],
    color_property: str = "tortuosity",
) -> "napari.types.LayerDataTuple | None":  # noqa: F821, UP037
    """Build one combined branch-paths layer spanning all objects.

    Parameters
    ----------
    base_name : str
        Base name used for layer naming.
    objects : list[ObjectResult]
        Per-object results (from `AnalysisResult.objects`). Branch path
        coordinates are offset into global image coordinates using each
        object's own crop offset.
    image_shape : tuple[int, ...]
        Full image shape (e.g. `AnalysisResult.skeleton.shape`), used to
        scale edge_width to the image's own size - see
        `_proportional_edge_width`.
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
        "edge_width": _proportional_edge_width(image_shape),
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
    branch_layer: "napari.types.LayerDataTuple",  # noqa: F821, UP037
    base_name: str,
) -> "napari.types.LayerDataTuple":  # noqa: F821, UP037
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
) -> "napari.types.LayerDataTuple | None":  # noqa: F821, UP037
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
        **_POINT_SIZE_KWARGS,
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
) -> "napari.types.LayerDataTuple | None":  # noqa: F821, UP037
    """Create a points layer showing branch/end nodes (degree != 2), colored
    by degree.

    Pass-through nodes (degree == 2) are omitted from this layer - they
    just mark where a branch bends, not a true topological feature, so
    dropping them makes the layer read as a graph (endpoints and
    junctions only) alongside the branches layer, rather than every single
    skeleton-graph node. This is a display-only filter: the underlying
    ``node_records``/node CSV export are unaffected and still include
    every node, including pass-through ones.

    Parameters
    ----------
    base_name : str
        Base name for layer naming.
    node_records : list[dict[str, object]]
        Pre-computed node records (e.g. from `AnalysisResult.node_records`),
        already in global image coordinates and tagged with ``object_id``.
    """
    filtered = [r for r in node_records if r.get("degree") != 2]
    if not filtered:
        return None

    ndim = sum(1 for k in filtered[0] if k.startswith("coord_"))
    points = np.array(
        [tuple(r[f"coord_{d}"] for d in range(ndim)) for r in filtered], dtype=float
    )

    props = {
        k: [r[k] for r in filtered] for k in filtered[0] if not k.startswith("coord_")
    }

    meta = {
        "name": f"{base_name}_branch_and_end_nodes",
        "properties": props,
        "symbol": "disc",
        "size": 3,
        **_POINT_SIZE_KWARGS,
        "face_color": "degree",
        "face_colormap": "viridis",
        "opacity": 0.8,
    }
    return (points, meta, "points")
