"""Napari widget for mask analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, Label, PushButton
from maskel._io import save_analysis_outputs
from maskel.pipeline import analyze_segmentation_mask
from napari.layers import Layer, Shapes
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QFileDialog

from napari_maskel.napari_layers import extract_skeleton_layers

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
        super().__init__()
        self.viewer = napari_viewer
        self._output_dir: Path | None = None
        self._setup_ui()

    def _setup_ui(self):
        # ---------- extraction parameters (magicgui) ----------
        def _extraction_params(
            image: "napari.layers.Labels",  # noqa: F821, UP037
            extract_branches: bool = False,
            branch_color_property: str = "tortuosity",
            extract_branch_text: bool = False,
            extract_nodes: bool = False,
            extract_summary: bool = True,
            include_fractal: bool = False,
            include_mask_radius: bool = False,
            junction_cleanup: bool = False,
            cleanup_threshold_factor: float = 2.5,
            prune_spurs: bool = False,
            min_spur_length: float = 10.0,
            spur_iterations: int = 1,
            fill_holes: bool = False,
            closing_iterations: int = 0,
            max_hole_size: int = 0,
            show_preprocessed: bool = False,
        ) -> None:
            return None

        extraction_gui = magicgui(
            _extraction_params,
            image={"label": "Input segmentation (labels layer)"},
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
            show_preprocessed={"annotation": bool, "value": False},
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
        ) -> None:
            return None

        output_gui = magicgui(_output_params)

        self._extraction_gui = extraction_gui
        self._output_gui = output_gui
        self.image_widget = extraction_gui.image

        # ============================================================
        # Extraction Layers
        # ============================================================
        extraction_group = Container()
        extraction_group.label = "Extraction Layers"

        self.extract_branches_widget = extraction_gui.extract_branches
        self.extract_branches_widget.label = "Extract branches"

        self.branch_color_widget = extraction_gui.branch_color_property
        self.branch_color_widget.label = "Branch color by"
        self.branch_color_widget.enabled = False

        self.branch_color_warning = Label(value="⚠️ Requires Mask Radius")
        self.branch_color_warning.visible = False

        def _on_branches_toggle(enabled: bool | None = None) -> None:
            self.branch_color_widget.enabled = self.extract_branches_widget.value

        self.extract_branches_widget.changed.connect(_on_branches_toggle)

        def _update_branch_color_warning(*args) -> None:
            needs_radius = self.branch_color_widget.value in _RADIUS_REQUIRED_PROPS
            radius_off = not self.include_mask_radius_widget.value
            self.branch_color_warning.visible = needs_radius and radius_off

        self.branch_color_widget.changed.connect(_update_branch_color_warning)
        self.branch_color_widget.changed.connect(self._recolor_branch_layers)

        # connection to include_mask_radius_widget happens below
        # after that widget is created

        self.extract_branch_text_widget = extraction_gui.extract_branch_text
        self.extract_branch_text_widget.label = "Add branch labels"
        self.extract_branch_text_widget.enabled = self.extract_branches_widget.value

        def _on_branches_toggle_branch_text(*args) -> None:
            self.extract_branch_text_widget.enabled = self.extract_branches_widget.value

        self.extract_branches_widget.changed.connect(_on_branches_toggle_branch_text)

        self.extract_summary_widget = extraction_gui.extract_summary
        self.extract_summary_widget.label = "Extract summary statistics"

        self.extract_nodes_widget = extraction_gui.extract_nodes
        self.extract_nodes_widget.label = "Extract node features"

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
        cleanup_group.label = "Cleanup"

        self.fill_holes_widget = extraction_gui.fill_holes
        self.fill_holes_widget.label = "Fill holes in segmentation"

        self.max_hole_size_widget = extraction_gui.max_hole_size
        self.max_hole_size_widget.label = "Max hole size (pixels)"
        self.max_hole_size_widget.enabled = False

        def _on_fill_holes_toggle(enabled: bool | None = None) -> None:
            self.max_hole_size_widget.enabled = self.fill_holes_widget.value

        self.fill_holes_widget.changed.connect(_on_fill_holes_toggle)

        self.closing_iterations_widget = extraction_gui.closing_iterations
        self.closing_iterations_widget.label = "Closing iterations"

        self.junction_cleanup_widget = extraction_gui.junction_cleanup
        self.junction_cleanup_widget.label = "Collapse triangle junction artifacts"

        self.cleanup_threshold_widget = extraction_gui.cleanup_threshold_factor
        self.cleanup_threshold_widget.label = "Cleanup threshold factor"
        self.cleanup_threshold_widget.enabled = False

        def _on_junction_cleanup_toggle(enabled: bool | None = None) -> None:
            self.cleanup_threshold_widget.enabled = self.junction_cleanup_widget.value

        self.junction_cleanup_widget.changed.connect(_on_junction_cleanup_toggle)

        self.prune_spurs_widget = extraction_gui.prune_spurs
        self.prune_spurs_widget.label = "Prune short spur branches"

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

        self.show_preprocessed_widget = extraction_gui.show_preprocessed
        self.show_preprocessed_widget.label = "Show preprocessed binary layer"
        self.show_preprocessed_widget.enabled = False

        def _update_preprocessed_enabled(*args) -> None:
            self.show_preprocessed_widget.enabled = (
                self.fill_holes_widget.value or self.closing_iterations_widget.value > 0
            )

        self.fill_holes_widget.changed.connect(_update_preprocessed_enabled)
        self.closing_iterations_widget.changed.connect(_update_preprocessed_enabled)

        cleanup_group.append(self.fill_holes_widget)
        cleanup_group.append(self.max_hole_size_widget)
        cleanup_group.append(self.closing_iterations_widget)
        cleanup_group.append(self.junction_cleanup_widget)
        cleanup_group.append(self.cleanup_threshold_widget)
        cleanup_group.append(self.prune_spurs_widget)
        cleanup_group.append(self.min_spur_length_widget)
        cleanup_group.append(self.spur_iterations_widget)
        cleanup_group.append(self.show_preprocessed_widget)

        # ============================================================
        # Advanced Features
        # ============================================================
        advanced_group = Container()
        advanced_group.label = "Advanced Features"

        self.include_fractal_widget = extraction_gui.include_fractal

        advanced_group.append(self.include_fractal_widget)

        self.include_mask_radius_widget = extraction_gui.include_mask_radius

        self.include_mask_radius_widget.changed.connect(_update_branch_color_warning)

        advanced_group.append(self.include_mask_radius_widget)

        # ============================================================
        # Output Settings (CLI file export options)
        # ============================================================
        output_group = Container()
        output_group.label = "Output Settings"

        self.write_skeleton_npy_widget = output_gui.write_skeleton_npy
        self.write_skeleton_npy_widget.label = "Write skeleton (.npy)"

        self.write_skeleton_png_widget = output_gui.write_skeleton_png
        self.write_skeleton_png_widget.label = "Write skeleton (.png)"

        self.write_summary_csv_widget = output_gui.write_summary_csv
        self.write_summary_csv_widget.label = "Write summary CSV"

        self.write_radius_widget = output_gui.write_radius
        self.write_radius_widget.label = "Write radius matrix (.npy)"
        self.write_radius_widget.enabled = self.include_mask_radius_widget.value

        def _on_mask_radius_toggle_write_radius(*args) -> None:
            self.write_radius_widget.enabled = self.include_mask_radius_widget.value

        self.include_mask_radius_widget.changed.connect(
            _on_mask_radius_toggle_write_radius
        )

        self.write_summary_csv_widget.enabled = self.extract_summary_widget.value

        def _on_summary_toggle_write_summary_csv(*args) -> None:
            self.write_summary_csv_widget.enabled = self.extract_summary_widget.value

        self.extract_summary_widget.changed.connect(
            _on_summary_toggle_write_summary_csv
        )

        self.write_branch_csv_widget = output_gui.write_branch_csv
        self.write_branch_csv_widget.label = "Write branch CSV"
        self.write_branch_csv_widget.enabled = self.extract_branches_widget.value

        def _on_branches_toggle_write_branch_csv(*args) -> None:
            self.write_branch_csv_widget.enabled = self.extract_branches_widget.value

        self.extract_branches_widget.changed.connect(
            _on_branches_toggle_write_branch_csv
        )

        self.write_node_csv_widget = output_gui.write_node_csv
        self.write_node_csv_widget.label = "Write node CSV"
        self.write_node_csv_widget.enabled = self.extract_nodes_widget.value

        def _on_nodes_toggle_write_node_csv(*args) -> None:
            self.write_node_csv_widget.enabled = self.extract_nodes_widget.value

        self.extract_nodes_widget.changed.connect(_on_nodes_toggle_write_node_csv)

        self.write_graphml_widget = output_gui.write_graphml
        self.write_graphml_widget.label = "Write graph (.graphml)"

        output_group.append(self.write_skeleton_npy_widget)
        output_group.append(self.write_skeleton_png_widget)
        output_group.append(self.write_summary_csv_widget)
        output_group.append(self.write_branch_csv_widget)
        output_group.append(self.write_node_csv_widget)
        output_group.append(self.write_radius_widget)
        output_group.append(self.write_graphml_widget)

        # ============================================================
        # Output Directory
        # ============================================================
        outdir_group = Container()
        outdir_group.label = "Output Directory"

        self.select_outdir_btn = PushButton(text="Select Output Directory...")
        self.select_outdir_btn.clicked.connect(self._on_select_output_dir)

        outdir_group.append(self.select_outdir_btn)

        # ============================================================
        # Configuration Management
        # ============================================================
        config_group = Container()
        config_group.label = "Configuration"

        self.load_btn = PushButton(text="Load Config")
        self.load_btn.clicked.connect(self._on_load_config)

        self.save_btn = PushButton(text="Save Config")
        self.save_btn.clicked.connect(self._on_save_config)

        config_group.append(self.load_btn)
        config_group.append(self.save_btn)

        # ============================================================
        # Analyze Button
        # ============================================================
        self.analyze_btn = PushButton(text="Analyze Mask")
        self.analyze_btn.clicked.connect(self._on_analyze)

        # ============================================================
        # Assemble widget
        # ============================================================
        self.append(self.image_widget)
        self.append(extraction_group)
        self.append(cleanup_group)
        self.append(advanced_group)
        self.append(output_group)
        self.append(outdir_group)
        self.append(config_group)
        self.append(self.analyze_btn)

    # ------------------------------------------------------------------
    # Config get / set (full PipelineConfig)
    # ------------------------------------------------------------------

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
            ),
            output=OutputConfig(
                write_skeleton_npy=self.write_skeleton_npy_widget.value,
                write_skeleton_png=self.write_skeleton_png_widget.value,
                write_summary_csv=self.write_summary_csv_widget.value,
                write_branch_csv=self.write_branch_csv_widget.value,
                write_node_csv=self.write_node_csv_widget.value,
                write_radius=self.write_radius_widget.value,
                write_graphml=self.write_graphml_widget.value,
            ),
        )

    def _set_pipeline_config(self, config: PipelineConfig) -> None:
        e = config.extraction
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
            show_info("Please select an image layer")
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
