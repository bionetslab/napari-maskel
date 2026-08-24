"""Napari widget for mask analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui
from magicgui.widgets import CheckBox, Container, Label, PushButton
from maskel._io import save_analysis_outputs
from maskel.pipeline import analyze_segmentation_mask
from napari.layers import Layer, Shapes
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QFileDialog, QLabel, QSizePolicy
from superqt import QCollapsible

from napari_maskel.napari_layers import extract_skeleton_layers, parse_spacing_input

if TYPE_CHECKING:
    # These imports are only used for annotations and are therefore
    # guarded by TYPE_CHECKING to avoid runtime import-time coupling.
    from napari.layers import Labels  # noqa: F401

from maskel.config import (
    COLORABLE_BRANCH_PROPERTIES,
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
    save_pipeline_config,
)

_LOGO_PATH = Path(__file__).parent / "resources" / "logo.png"
_LOGO_DISPLAY_HEIGHT = 108

_BRAND_COLOR = "#CD53A1"
# Overrides napari's own (blue-accented) theme qss for just this widget's
# checkboxes and QCollapsible section headers. Set on the widget's own
# native container rather than the QApplication, since a widget's own
# stylesheet takes precedence over an ancestor-applied one for matching
# selectors - so this only recolors maskel's controls, not napari's.
# QCollapsible's header is a checkable QPushButton (see _make_collapsible);
# its `:checked` state is "expanded", which is the default for every
# section here, so in practice this covers what reads as the header color.
_BRAND_STYLESHEET = f"""
QCheckBox::indicator:checked {{
    background-color: {_BRAND_COLOR};
    border: 1px solid {_BRAND_COLOR};
}}
QPushButton:checked {{
    background-color: {_BRAND_COLOR};
}}
"""

_RADIUS_REQUIRED_PROPS = {
    "mean_radius",
    "std_radius",
    "min_radius",
    "max_radius",
    "mean_diameter",
    "std_diameter",
    "min_diameter",
    "max_diameter",
    "volume",
    "surface_area",
}


class MaskAnalysisWidget(Container):
    """Analysis configuration widget."""

    _CONFIG_FILTER = "JSON Files (*.json);;All Files (*)"

    def __init__(self, napari_viewer):
        super().__init__(scrollable=True)
        # napari's add_dock_widget always embeds `widget.native` verbatim,
        # but magicgui's own `.native` deliberately returns the *unwrapped*
        # content widget when scrollable=True ("this is the widget that
        # contains the layout, and not any parent widget ... used to enable
        # scroll bars" - see magicgui's own docstring). Without this
        # override, napari would silently discard the scroll-area wrapper
        # and vertical scrolling would never actually take effect once
        # docked. `.root_native_widget` is the wrapper; keep a handle to
        # the real content widget (the one with the actual child layout)
        # for our own use below, since our own `native` override shadows it.
        self._content_native = super().native
        self.viewer = napari_viewer
        self._output_dir: Path | None = None
        self._setup_ui()

    @property
    def native(self):
        """The widget napari should dock - the scroll-area wrapper, not the
        bare content widget (see the comment in ``__init__``)."""
        return self.root_native_widget

    @staticmethod
    def _build_logo_label() -> QLabel:
        """A small, left-aligned banner showing the maskel logo.

        Built as a raw ``QLabel`` inserted directly into the content
        widget's layout (like the ``QCollapsible`` groups below), not
        through ``Container.append()`` - see the note on that in this
        file's module docstring area / CLAUDE.md.
        """
        label = QLabel()
        pixmap = QPixmap(str(_LOGO_PATH))
        label.setPixmap(
            pixmap.scaledToHeight(_LOGO_DISPLAY_HEIGHT, Qt.SmoothTransformation)
        )
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def _setup_ui(self):
        # ---------- extraction parameters (magicgui) ----------
        def _extraction_params(
            image: "napari.layers.Labels",  # noqa: F821, UP037
            spacing: str = "",
            extract_branches: bool = False,
            branch_color_property: str = "tortuosity",
            extract_branch_text: bool = False,
            extract_nodes: bool = False,
            extract_summary: bool = True,
            include_fractal: bool = False,
            include_mask_radius: bool = False,
            show_preprocessed: bool = False,
            junction_cleanup: bool = False,
            cleanup_threshold_factor: float = 2.5,
            prune_spurs: bool = False,
            min_spur_length: float = 10.0,
            spur_iterations: int = 1,
            fill_holes: bool = False,
            closing_iterations: int = 0,
            max_hole_size: int = 0,
        ) -> None:
            return None

        extraction_gui = magicgui(
            _extraction_params,
            image={"label": "Input segmentation (labels layer)"},
            spacing={
                "annotation": str,
                "value": "",
                "label": "Spacing (comma-separated, e.g. 1.0,1.0)",
            },
            extract_branches={"annotation": bool, "value": False},
            branch_color_property={
                "annotation": str,
                "value": "tortuosity",
                "choices": COLORABLE_BRANCH_PROPERTIES,
                "widget_type": "ComboBox",
            },
            extract_branch_text={"annotation": bool, "value": False},
            extract_summary={"annotation": bool, "value": True},
            include_fractal={
                "annotation": bool,
                "value": False,
                "label": "Fractal dimension",
            },
            include_mask_radius={
                "annotation": bool,
                "value": False,
                "label": "Radius features",
            },
            show_preprocessed={"annotation": bool, "value": False},
            junction_cleanup={"annotation": bool, "value": False},
            cleanup_threshold_factor={
                "annotation": float,
                "value": 2.5,
                "widget_type": "FloatSpinBox",
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
            },
            prune_spurs={"annotation": bool, "value": False},
            min_spur_length={
                "annotation": float,
                "value": 10.0,
                "widget_type": "FloatSpinBox",
                "min": 0.0,
                "max": 1000.0,
                "step": 1.0,
            },
            spur_iterations={
                "annotation": int,
                "value": 1,
                "widget_type": "SpinBox",
                "min": 1,
                "max": 100,
                "step": 1,
            },
            fill_holes={"annotation": bool, "value": False},
            closing_iterations={
                "annotation": int,
                "value": 0,
                "widget_type": "SpinBox",
                "min": 0,
                "max": 10,
                "step": 1,
            },
            max_hole_size={
                "annotation": int,
                "value": 0,
                "widget_type": "SpinBox",
                "min": 0,
                "max": 100000,
                "step": 100,
            },
        )

        # ---------- output parameters (magicgui) ----------
        def _output_params(
            write_skeleton_npy: bool = True,
            write_skeleton_png: bool = False,
            write_summary_csv: bool = True,
            write_branch_csv: bool = False,
            write_node_csv: bool = False,
            write_radius: bool = False,
            write_graphml: bool = False,
            write_networkx_graph: bool = False,
        ) -> None:
            return None

        output_gui = magicgui(_output_params)

        self._extraction_gui = extraction_gui
        self._output_gui = output_gui
        self.image_widget = extraction_gui.image

        # ============================================================
        # Physical Spacing
        # ============================================================
        spacing_group = Container()

        self.image_shape_label = Label(value="")
        self._prepare_wrapping_label(self.image_shape_label)

        def _update_image_shape_label(*args) -> None:
            img = self.image_widget.value
            self.image_shape_label.value = (
                f"Image shape: {img.data.shape} (enter spacing in this axis order)"
                if img is not None
                else ""
            )

        self.image_widget.changed.connect(_update_image_shape_label)

        self.spacing_widget = extraction_gui.spacing

        self.spacing_warning = Label(value="")
        self._prepare_wrapping_label(self.spacing_warning)
        self.spacing_warning.visible = False

        def _default_spacing_from_layer(*args) -> None:
            """Napari layers always have a `.scale` of length ndim
            (defaulting to all 1.0) - the natural default for this field,
            since napari itself has no convenient numeric field to *set*
            `.scale` (only an imprecise drag-based Transform tool, or the
            console)."""
            img = self.image_widget.value
            if img is None:
                return
            scale = getattr(img, "scale", None)
            if scale is None:
                return
            self.spacing_widget.value = ",".join(f"{s:g}" for s in scale)

        def _update_spacing_warning(*args) -> None:
            img = self.image_widget.value
            if img is None:
                self.spacing_warning.visible = False
                return
            _, error = parse_spacing_input(self.spacing_widget.value, img.data.ndim)
            self.spacing_warning.value = f"⚠️ {error}" if error else ""
            self.spacing_warning.visible = error is not None

        self.image_widget.changed.connect(_default_spacing_from_layer)
        self.image_widget.changed.connect(_update_spacing_warning)
        self.spacing_widget.changed.connect(_update_spacing_warning)

        def _sync_image_dependent_widgets(*args) -> None:
            _update_image_shape_label()
            _default_spacing_from_layer()
            _update_spacing_warning()

        self._sync_image_dependent_widgets = _sync_image_dependent_widgets

        # `image_widget` may already have a layer auto-selected at this
        # point (magicgui picks one during construction, before any of the
        # above .changed connections existed to react to it) - run this
        # once now so that layer's shape/default-spacing/warning aren't
        # left stale just because no *change* event ever fired for it.
        self._sync_image_dependent_widgets()

        spacing_group.append(self.image_shape_label)
        spacing_group.append(self.spacing_widget)
        spacing_group.append(self.spacing_warning)

        # ============================================================
        # Extraction Layers
        # ============================================================
        extraction_group = Container()

        self.extract_branches_widget = extraction_gui.extract_branches
        self._set_checkbox_text(self.extract_branches_widget, "Extract branch features")

        self.branch_color_widget = extraction_gui.branch_color_property
        self.branch_color_widget.label = "Branch color by"
        self.branch_color_widget.enabled = False

        self.branch_color_warning = Label(value="⚠️ Requires Mask Radius")
        self.branch_color_warning.visible = False

        def _on_branches_toggle(enabled: bool | None = None) -> None:
            self.branch_color_widget.enabled = self.extract_branches_widget.value

        self.extract_branches_widget.changed.connect(_on_branches_toggle)
        self.extract_branches_widget.changed.connect(self._reconcile_dependent_widgets)

        def _update_branch_color_warning(*args) -> None:
            needs_radius = self.branch_color_widget.value in _RADIUS_REQUIRED_PROPS
            radius_off = not self.include_mask_radius_widget.value
            self.branch_color_warning.visible = needs_radius and radius_off

        self.branch_color_widget.changed.connect(_update_branch_color_warning)
        self.branch_color_widget.changed.connect(self._recolor_branch_layers)

        # connection to include_mask_radius_widget happens below
        # after that widget is created

        self.extract_branch_text_widget = extraction_gui.extract_branch_text
        self._set_checkbox_text(self.extract_branch_text_widget, "Add branch labels")

        self.extract_summary_widget = extraction_gui.extract_summary
        self._set_checkbox_text(
            self.extract_summary_widget, "Extract object-level features"
        )
        self.extract_summary_widget.changed.connect(self._reconcile_dependent_widgets)

        self.extract_nodes_widget = extraction_gui.extract_nodes
        self._set_checkbox_text(self.extract_nodes_widget, "Extract node features")
        self.extract_nodes_widget.changed.connect(self._reconcile_dependent_widgets)

        extraction_group.append(self.extract_branches_widget)
        extraction_group.append(self.branch_color_widget)
        extraction_group.append(self.branch_color_warning)
        extraction_group.append(self.extract_branch_text_widget)
        extraction_group.append(self.extract_summary_widget)
        extraction_group.append(self.extract_nodes_widget)

        # ============================================================
        # Cleanup
        # ============================================================
        cleanup_group = Container()

        self.fill_holes_widget = extraction_gui.fill_holes
        self._set_checkbox_text(self.fill_holes_widget, "Fill holes in mask")

        self.max_hole_size_widget = extraction_gui.max_hole_size
        self.max_hole_size_widget.label = "Max hole size (pixels)"
        self.max_hole_size_widget.enabled = False

        def _on_fill_holes_toggle(enabled: bool | None = None) -> None:
            self.max_hole_size_widget.enabled = self.fill_holes_widget.value

        self.fill_holes_widget.changed.connect(_on_fill_holes_toggle)
        self.fill_holes_widget.changed.connect(self._reconcile_dependent_widgets)

        self.closing_iterations_widget = extraction_gui.closing_iterations
        self.closing_iterations_widget.label = "Closing iterations"
        self.closing_iterations_widget.changed.connect(
            self._reconcile_dependent_widgets
        )

        self.show_preprocessed_widget = extraction_gui.show_preprocessed
        self._set_checkbox_text(self.show_preprocessed_widget, "Show preprocessed mask")

        self.junction_cleanup_widget = extraction_gui.junction_cleanup
        self._set_checkbox_text(
            self.junction_cleanup_widget, "Skeleton junction cleanup"
        )

        self.cleanup_threshold_widget = extraction_gui.cleanup_threshold_factor
        self.cleanup_threshold_widget.label = "Cleanup threshold factor"
        self.cleanup_threshold_widget.enabled = False

        def _on_junction_cleanup_toggle(enabled: bool | None = None) -> None:
            self.cleanup_threshold_widget.enabled = self.junction_cleanup_widget.value

        self.junction_cleanup_widget.changed.connect(_on_junction_cleanup_toggle)

        self.prune_spurs_widget = extraction_gui.prune_spurs
        self._set_checkbox_text(self.prune_spurs_widget, "Prune skeleton spurs")

        self.min_spur_length_widget = extraction_gui.min_spur_length
        self.min_spur_length_widget.label = "Min spur length (pixels)"
        self.min_spur_length_widget.enabled = False

        self.spur_iterations_widget = extraction_gui.spur_iterations
        self.spur_iterations_widget.label = "Spur iterations"
        self.spur_iterations_widget.enabled = False

        def _on_prune_spurs_toggle(enabled: bool | None = None) -> None:
            self.min_spur_length_widget.enabled = self.prune_spurs_widget.value
            self.spur_iterations_widget.enabled = self.prune_spurs_widget.value

        self.prune_spurs_widget.changed.connect(_on_prune_spurs_toggle)

        cleanup_group.append(self.fill_holes_widget)
        cleanup_group.append(self.max_hole_size_widget)
        cleanup_group.append(self.closing_iterations_widget)
        cleanup_group.append(self.show_preprocessed_widget)
        cleanup_group.append(self.junction_cleanup_widget)
        cleanup_group.append(self.cleanup_threshold_widget)
        cleanup_group.append(self.prune_spurs_widget)
        cleanup_group.append(self.min_spur_length_widget)
        cleanup_group.append(self.spur_iterations_widget)

        # ============================================================
        # Advanced Features
        # ============================================================
        advanced_group = Container()

        self.include_fractal_widget = extraction_gui.include_fractal

        advanced_group.append(self.include_fractal_widget)

        self.include_mask_radius_widget = extraction_gui.include_mask_radius

        self.include_mask_radius_widget.changed.connect(_update_branch_color_warning)
        self.include_mask_radius_widget.changed.connect(
            self._reconcile_dependent_widgets
        )

        advanced_group.append(self.include_mask_radius_widget)

        # ============================================================
        # Output Settings (CLI file export options)
        # ============================================================
        output_group = Container()

        self.write_skeleton_npy_widget = output_gui.write_skeleton_npy
        self._set_checkbox_text(self.write_skeleton_npy_widget, "Write skeleton (.npy)")

        self.write_skeleton_png_widget = output_gui.write_skeleton_png
        self._set_checkbox_text(self.write_skeleton_png_widget, "Write skeleton (.png)")

        self.write_skeleton_png_warning = Label(
            value="⚠️ 3D image selected: PNG export will be skipped"
        )
        self.write_skeleton_png_warning.visible = False

        def _update_skeleton_png_warning(*args) -> None:
            img = self.image_widget.value
            is_3d = img is not None and img.data.ndim == 3
            self.write_skeleton_png_warning.visible = (
                self.write_skeleton_png_widget.value and is_3d
            )

        self.write_skeleton_png_widget.changed.connect(_update_skeleton_png_warning)
        self.image_widget.changed.connect(_update_skeleton_png_warning)

        self.write_summary_csv_widget = output_gui.write_summary_csv
        self._set_checkbox_text(self.write_summary_csv_widget, "Write summary csv")

        self.write_radius_widget = output_gui.write_radius
        self._set_checkbox_text(self.write_radius_widget, "Write radius matrix (.npy)")

        self.write_branch_csv_widget = output_gui.write_branch_csv
        self._set_checkbox_text(self.write_branch_csv_widget, "Write branch csv")

        self.write_node_csv_widget = output_gui.write_node_csv
        self._set_checkbox_text(self.write_node_csv_widget, "Write node csv")

        self.write_graphml_widget = output_gui.write_graphml
        self._set_checkbox_text(self.write_graphml_widget, "Write graph (.graphml)")

        self.write_networkx_graph_widget = output_gui.write_networkx_graph
        self._set_checkbox_text(
            self.write_networkx_graph_widget, "Write networkx graph (.pkl)"
        )

        self._write_option_widgets = (
            self.write_skeleton_npy_widget,
            self.write_skeleton_png_widget,
            self.write_summary_csv_widget,
            self.write_branch_csv_widget,
            self.write_node_csv_widget,
            self.write_radius_widget,
            self.write_graphml_widget,
            self.write_networkx_graph_widget,
        )

        self.select_outdir_btn = PushButton(text="Select output directory...")
        self.select_outdir_btn.clicked.connect(self._on_select_output_dir)

        self.output_dir_warning = Label(value="⚠️ Please select output directory")
        self.output_dir_warning.visible = False

        for w in self._write_option_widgets:
            w.changed.connect(self._update_output_dir_controls)
        self._update_output_dir_controls()
        self._reconcile_dependent_widgets()

        output_group.append(self.write_skeleton_npy_widget)
        output_group.append(self.write_skeleton_png_widget)
        output_group.append(self.write_skeleton_png_warning)
        output_group.append(self.write_summary_csv_widget)
        output_group.append(self.write_branch_csv_widget)
        output_group.append(self.write_node_csv_widget)
        output_group.append(self.write_radius_widget)
        output_group.append(self.write_graphml_widget)
        output_group.append(self.write_networkx_graph_widget)
        output_group.append(self.select_outdir_btn)
        output_group.append(self.output_dir_warning)

        # ============================================================
        # Configuration Management
        # ============================================================
        config_group = Container()

        self.load_btn = PushButton(text="Load recipe")
        self.load_btn.clicked.connect(self._on_load_config)

        self.save_btn = PushButton(text="Save recipe")
        self.save_btn.clicked.connect(self._on_save_config)

        config_group.append(self.load_btn)
        config_group.append(self.save_btn)

        # ============================================================
        # Analyze Button
        # ============================================================
        self.analyze_btn = PushButton(text="Analyze mask")
        self.analyze_btn.clicked.connect(self._on_analyze)

        # ============================================================
        # Assemble widget
        # ============================================================
        # self.append() inserts at a position computed from magicgui's own
        # internal widget count, and QCollapsible is a raw Qt widget outside
        # that bookkeeping - mixing the two desyncs the count and silently
        # reorders every later self.append() call. self.image_widget is the
        # sole exception: it's the very first widget added, so self.append()
        # here is safe (there's nothing before it to desync against) and it
        # preserves the "Input segmentation (labels layer)" label, which
        # self.append() renders by wrapping non-button widgets in an
        # internal _LabeledWidget - something a raw .native insert would
        # silently drop. Everything after it (each group, wrapped in its own
        # collapsible and expanded by default, plus the final Analyze
        # button) goes straight into the content widget's real Qt layout.
        self.append(self.image_widget)
        content_layout = self._content_native.layout()
        # Inserted after the append() above (not before) - see the comment
        # on desync risk just above: a raw insert done *before* self.append()
        # would still be pushed down to index 1 by append()'s own position
        # bookkeeping, which is unaware of it.
        content_layout.insertWidget(0, self._build_logo_label())

        self._spacing_collapsible = self._make_collapsible(
            spacing_group, "Physical spacing"
        )
        self._extraction_collapsible = self._make_collapsible(
            extraction_group, "Extraction layers"
        )
        self._cleanup_collapsible = self._make_collapsible(cleanup_group, "Cleanup")
        self._advanced_collapsible = self._make_collapsible(
            advanced_group, "Advanced features"
        )
        self._output_collapsible = self._make_collapsible(
            output_group, "Output settings"
        )
        self._config_collapsible = self._make_collapsible(config_group, "Recipe")

        for collapsible in (
            self._spacing_collapsible,
            self._extraction_collapsible,
            self._cleanup_collapsible,
            self._advanced_collapsible,
            self._output_collapsible,
            self._config_collapsible,
        ):
            content_layout.addWidget(collapsible)

        content_layout.addWidget(self.analyze_btn.native)

        self._content_native.setStyleSheet(_BRAND_STYLESHEET)

    @staticmethod
    def _make_collapsible(group: Container, title: str) -> QCollapsible:
        """Wrap a magicgui ``Container`` in a ``QCollapsible``, expanded by default.

        The group's own reactive wiring (``.enabled``/``.value`` toggling on its
        child widgets) is untouched - it operates on the widgets directly and
        doesn't depend on where the group ends up in the Qt widget tree.
        """
        collapsible = QCollapsible(title=title)
        collapsible.addWidget(group.native)
        collapsible.expand(animate=False)
        return collapsible

    @staticmethod
    def _set_checkbox_text(widget: CheckBox, text: str) -> None:
        """Set a magicgui ``CheckBox``'s displayed text.

        Unlike most widget types - which pair with a separate ``QLabel``
        reflecting ``.label``, rendered by the parent ``Container`` - a
        ``CheckBox`` shows its own caption directly on the native Qt
        control, set once at construction from either an explicit
        ``"label"`` option or (absent one) the parameter name. Setting
        ``.label`` afterward, the pattern used everywhere else in this
        file, updates that Python-level attribute but never touches what's
        actually drawn on screen - the checkbox keeps showing its original
        auto-derived text (e.g. "junction cleanup" instead of "Skeleton
        junction cleanup") regardless. ``.text`` is the separate attribute
        that actually is wired to the native control; ``.label`` is set
        too since magicgui uses it internally (e.g. to align label widths
        across a container), so the two shouldn't be left inconsistent.
        """
        widget.label = text
        widget.text = text

    @staticmethod
    def _prepare_wrapping_label(label: Label) -> None:
        """Let a magicgui ``Label`` wrap and shrink instead of growing its
        parent to fit a long one-line message.

        magicgui's ``Label`` backend hard-codes
        ``QSizePolicy.Fixed`` on both axes at construction time - the
        label is always sized to its own unconstrained ``sizeHint()``,
        never shrunk by the layout, regardless of ``setWordWrap()``. For a
        label fed from an f-string with unbounded content (an image shape
        tuple, a validation error), that fixed sizeHint is a full one-line
        width, which silently grows the whole widget past the dock panel's
        width rather than wrapping - the overflow just gets clipped
        off-screen. Overriding the horizontal policy to ``Ignored`` tells
        the layout to disregard that sizeHint entirely and give the label
        whatever width the container actually has, which combined with
        word wrap makes it wrap within that width instead. Left-aligned
        explicitly since the now-wider-than-its-text box would otherwise
        leave the default alignment ambiguous to a reader of this code.
        """
        native = label.native
        native.setWordWrap(True)
        native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        native.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    # ------------------------------------------------------------------
    # Config get / set (full PipelineConfig)
    # ------------------------------------------------------------------

    def _get_current_spacing(self) -> tuple[float, ...] | None:
        """Re-parse the spacing field against the currently selected image's
        dimensionality at read time, rather than trusting cached state -
        falls back to ``None`` (isotropic) with no image selected or on an
        invalid value, same "warn/fall back, don't crash" policy as the
        CLI's ``--spacing``, surfaced here as ``spacing_warning`` instead of
        a stderr print."""
        img = self.image_widget.value
        if img is None:
            return None
        spacing, _ = parse_spacing_input(self.spacing_widget.value, img.data.ndim)
        return spacing

    def _get_current_pipeline_config(self) -> PipelineConfig:
        junction_cleanup = self.junction_cleanup_widget.value
        return PipelineConfig(
            extraction=ExtractionConfig(
                branches=self.extract_branches_widget.value,
                branch_color_property=self.branch_color_widget.value,
                branch_text=self.extract_branch_text_widget.value,
                nodes=self.extract_nodes_widget.value,
                summary=self.extract_summary_widget.value,
                fractal_dimension=self.include_fractal_widget.value,
                mask_radius=self.include_mask_radius_widget.value,
                junction_cleanup=junction_cleanup,
                cleanup_threshold_factor=self.cleanup_threshold_widget.value,
                prune_spurs=self.prune_spurs_widget.value,
                min_spur_length=self.min_spur_length_widget.value,
                spur_iterations=self.spur_iterations_widget.value,
                fill_holes=self.fill_holes_widget.value,
                closing_iterations=self.closing_iterations_widget.value,
                max_hole_size=self.max_hole_size_widget.value,
                show_preprocessed=self.show_preprocessed_widget.value,
                spacing=self._get_current_spacing(),
            ),
            output=OutputConfig(
                write_skeleton_npy=self.write_skeleton_npy_widget.value,
                write_skeleton_png=self.write_skeleton_png_widget.value,
                write_summary_csv=self.write_summary_csv_widget.value,
                write_branch_csv=self.write_branch_csv_widget.value,
                write_node_csv=self.write_node_csv_widget.value,
                write_radius=self.write_radius_widget.value,
                write_graphml=self.write_graphml_widget.value,
                write_networkx_graph=self.write_networkx_graph_widget.value,
            ),
        )

    def _set_pipeline_config(self, config: PipelineConfig) -> None:
        e = config.extraction
        self.spacing_widget.value = (
            ",".join(str(s) for s in e.spacing) if e.spacing is not None else ""
        )
        self.extract_branches_widget.value = e.branches
        self.branch_color_widget.value = e.branch_color_property
        self.extract_branch_text_widget.value = e.branch_text
        self.extract_nodes_widget.value = e.nodes
        self.extract_summary_widget.value = e.summary
        self.include_fractal_widget.value = e.fractal_dimension
        self.include_mask_radius_widget.value = e.mask_radius
        self.junction_cleanup_widget.value = e.junction_cleanup
        self.cleanup_threshold_widget.value = e.cleanup_threshold_factor
        self.prune_spurs_widget.value = e.prune_spurs
        self.min_spur_length_widget.value = e.min_spur_length
        self.spur_iterations_widget.value = e.spur_iterations
        self.fill_holes_widget.value = e.fill_holes
        self.closing_iterations_widget.value = e.closing_iterations
        self.max_hole_size_widget.value = e.max_hole_size
        self.show_preprocessed_widget.value = e.show_preprocessed

        o = config.output
        self.write_skeleton_npy_widget.value = o.write_skeleton_npy
        self.write_skeleton_png_widget.value = o.write_skeleton_png
        self.write_summary_csv_widget.value = o.write_summary_csv
        self.write_branch_csv_widget.value = o.write_branch_csv
        self.write_node_csv_widget.value = o.write_node_csv
        self.write_radius_widget.value = o.write_radius
        self.write_graphml_widget.value = o.write_graphml
        self.write_networkx_graph_widget.value = o.write_networkx_graph

        # A loaded config file can itself contain an inconsistent
        # combination (e.g. write_radius=true with mask_radius=false) -
        # reconcile after setting the raw values so the widget never
        # displays one, rather than only preventing it going forward.
        self._reconcile_dependent_widgets()
        self._update_output_dir_controls()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_load_config(self) -> None:
        """Load configuration from file."""
        try:
            config_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Pipeline Configuration",
                "",
                self._CONFIG_FILTER,
            )
            if not config_path:
                return

            pipeline_config = load_pipeline_config(Path(config_path))
            self._set_pipeline_config(pipeline_config)
            show_info("Configuration loaded")
        except (ValueError, OSError) as e:
            show_error(f"Failed to load config: {e}")

    def _on_save_config(self) -> None:
        """Save current configuration to file."""
        try:
            pipeline_config = self._get_current_pipeline_config()
            config_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Pipeline Configuration",
                "",
                self._CONFIG_FILTER,
            )
            if not config_path:
                return

            save_pipeline_config(pipeline_config, Path(config_path))
            show_info(f"Configuration saved to {config_path}")
        except (ValueError, OSError) as e:
            show_error(f"Failed to save config: {e}")

    def _on_select_output_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(None, "Select Output Directory")
        if dir_path:
            self._output_dir = Path(dir_path)
            self.select_outdir_btn.text = str(self._output_dir)
            self._update_output_dir_controls()

    def _update_output_dir_controls(self, *args) -> None:
        """Keep the output-directory button and its warning in sync with
        whether any write option is active. The button is disabled when
        none is (nothing would be saved, so there's nothing to pick a
        directory for); the warning shows when one is active but no
        directory has been selected yet, since results would otherwise
        silently never be saved. ``self._output_dir`` is a plain attribute
        (not a magicgui widget), so it has no ``.changed`` signal of its
        own - this is called directly from ``_on_select_output_dir``
        instead. Selecting a directory while every write option happens to
        be off, then re-enabling one later, is unaffected: the previously
        selected path is preserved either way, only the button's enabled
        state and the warning change."""
        any_write_active = any(w.value for w in self._write_option_widgets)
        self.select_outdir_btn.enabled = any_write_active
        self.output_dir_warning.visible = any_write_active and self._output_dir is None

    def _reconcile_dependent_widgets(self, *args) -> None:
        """Keep every checkbox that's only meaningful while another
        checkbox is on in sync with that parent: disabled, and - unlike a
        plain ``.enabled`` toggle, which would otherwise leave it checked
        while grayed out - explicitly unchecked too. Without this, a
        config saved while e.g. "Radius features" was on and then turned
        back off would still have "Write radius matrix" sitting checked,
        producing a saved JSON with an inconsistent combination like
        ``write_radius=true, mask_radius=false`` that silently writes
        nothing. Connected to every parent's ``.changed`` below, and also
        called once after ``_set_pipeline_config`` sets raw values from a
        loaded config, so a config file with that same inconsistency gets
        corrected on load rather than reproduced in the widget."""
        for parent, dependent in (
            (self.extract_branches_widget, self.extract_branch_text_widget),
            (self.extract_branches_widget, self.write_branch_csv_widget),
            (self.include_mask_radius_widget, self.write_radius_widget),
            (self.extract_summary_widget, self.write_summary_csv_widget),
            (self.extract_nodes_widget, self.write_node_csv_widget),
        ):
            dependent.enabled = parent.value
            if not parent.value:
                dependent.value = False

        preprocessing_active = (
            self.fill_holes_widget.value or self.closing_iterations_widget.value > 0
        )
        self.show_preprocessed_widget.enabled = preprocessing_active
        if not preprocessing_active:
            self.show_preprocessed_widget.value = False

    def _recolor_branch_layers(self, *args) -> None:
        """Recolor existing branch layers without re-running analysis."""
        prop = self.branch_color_widget.value

        for layer in self.viewer.layers:
            if not isinstance(layer, Shapes) or not layer.name.endswith("_branches"):
                continue

            props = layer.properties
            values = props.get(prop) if props is not None else None
            if values is not None:
                numeric = np.asarray(values, dtype=float)
                finite = numeric[np.isfinite(numeric)]
                if finite.size > 1 and float(np.min(finite)) < float(np.max(finite)):
                    vmin = float(np.min(finite))
                    vmax = float(np.max(finite))
                    layer.edge_color = prop
                    layer.edge_colormap = "turbo"
                    layer.edge_contrast_limits = (vmin, vmax)
                    continue

            layer.edge_color = "#30d5c8"

    def _on_analyze(self) -> None:
        """Execute analysis with current settings."""
        img = self.image_widget.value
        if img is None:
            show_info("Please select a label layer")
            return

        try:
            t0 = time.perf_counter()
            pipeline_config = self._get_current_pipeline_config()
            result = analyze_segmentation_mask(mask=img.data, config=pipeline_config)
            elapsed = time.perf_counter() - t0

            n_fg = int((img.data > 0).sum())
            n_skel = int(result.skeleton.sum())
            show_info(f"Analysis: {n_fg} → {n_skel} skeleton pixels in {elapsed:.3f}s")

            # -- optional: show preprocessed binary layer ----------------
            if (
                pipeline_config.extraction.show_preprocessed
                and result.preprocessed_binary is not None
            ):
                self.viewer.add_layer(
                    Layer.create(
                        result.preprocessed_binary,
                        {"name": f"{img.name}_preprocessed"},
                        "labels",
                    )
                )

            layers = extract_skeleton_layers(
                result, img.name, config=pipeline_config.extraction
            )
            for data, meta, layer_type in layers:
                try:
                    layer = Layer.create(data, meta, layer_type)
                    self.viewer.add_layer(layer)
                except Exception as e:  # noqa: BLE001
                    show_info(
                        f"Failed to add layer {meta.get('name', '<unnamed>')}: {e}"
                    )

            # -- save results to disk if output directory is set ----------
            if self._output_dir is not None:
                save_analysis_outputs(
                    self._output_dir, img.name, result, pipeline_config.output
                )
                show_info(f"Results saved to {self._output_dir / img.name}")

        except (ValueError, RuntimeError, OSError) as e:
            show_error(f"Analysis failed: {e}")
