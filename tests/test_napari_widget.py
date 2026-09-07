"""Smoke tests for MaskAnalysisWidget construction, layout, and spacing-field wiring.

These run headless (QT_QPA_PLATFORM=offscreen) and use a plain mock in place
of a real napari.Viewer, since MaskAnalysisWidget.__init__ never calls
anything on the viewer during construction - it's only used later, in
_recolor_branch_layers/_on_analyze. That sidesteps needing napari's
make_napari_viewer fixture, which itself needs pytest-qt (not a project
dependency here).
"""

import json
import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from maskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    save_pipeline_config,
)
from napari.layers import Shapes
from qtpy.QtWidgets import QPushButton, QScrollArea

import napari_maskel._napari as napari_maskel_module
from napari_maskel._napari import MaskAnalysisWidget


@pytest.fixture
def widget():
    w = MaskAnalysisWidget(MagicMock())
    # Qt's `.visible` reflects real isVisible(), which is False for any
    # widget whose top-level window was never shown, regardless of
    # setVisible(True) calls on it - show() here so spacing_warning's
    # visibility toggling is actually observable in these headless tests.
    w.show()
    yield w
    w.close()


def _mock_layer(shape):
    layer = MagicMock()
    layer.data = MagicMock()
    layer.data.ndim = len(shape)
    layer.data.shape = shape
    layer.scale = tuple(1.0 for _ in shape)
    return layer


def _small_cross_mask():
    """A tiny real 2D binary mask, cheap enough for _on_analyze to process
    within a test - a cross gives at least one junction (so branch/node
    extraction has something non-trivial to do)."""
    img = np.zeros((16, 16), dtype=np.uint8)
    img[8, 2:14] = 1
    img[2:14, 8] = 1
    return img


def _real_image_layer(data, name="test_image", scale=None):
    """Like _mock_layer, but with a real ndarray for .data (and a real
    string .name) - needed wherever the pipeline actually runs on the
    layer's data, e.g. _on_analyze."""
    layer = MagicMock()
    layer.data = data
    layer.name = name
    layer.scale = scale or tuple(1.0 for _ in range(data.ndim))
    return layer


def _select_image(widget, layer):
    """Select *layer* on the image ComboBox.

    ``image_widget`` is a magicgui ``ComboBox`` (its choices normally come
    from the live napari viewer's layer list), so it rejects a value that
    isn't already one of its ``choices`` - add the mock layer to the
    choices first, same as a real viewer would when a layer is added.
    """
    widget.image_widget.choices = (layer,)
    widget.image_widget.value = layer


# -- scrollable / collapsible layout --------------------------------------


def test_constructs_and_is_scrollable(widget):
    # scrollable=True wraps the whole widget in a scroll area so it's never
    # clipped by napari's dock panel, regardless of content height.
    assert widget._scrollable is True


def test_brand_color_applied_to_checkboxes_and_headers(widget):
    from napari_maskel._napari import _BRAND_COLOR

    style = widget._content_native.styleSheet()
    assert "QCheckBox::indicator:checked" in style
    assert "QPushButton:checked" in style
    assert _BRAND_COLOR in style


def test_native_exposes_the_scroll_area_for_napari(widget):
    # napari's add_dock_widget embeds `widget.native` verbatim, but
    # magicgui's own `.native` deliberately returns the *unwrapped* content
    # widget when scrollable=True - without overriding it, napari would
    # discard the scroll area and vertical scrolling would never take
    # effect once docked.
    assert isinstance(widget.native, QScrollArea)
    from qtpy.QtCore import Qt

    assert (
        widget.native.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert (
        widget.native.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )


def test_all_groups_collapsible_and_expanded_by_default(widget):
    for name in (
        "_spacing_collapsible",
        "_extraction_collapsible",
        "_cleanup_collapsible",
        "_advanced_collapsible",
        "_output_collapsible",
        "_config_collapsible",
    ):
        assert getattr(widget, name).isExpanded() is True


def test_group_order_matches_original_layout(widget):
    # logo, image selector, then every group in its original relative
    # order, then the Analyze button last. Regression test for a bug where
    # mixing self.append() with raw layout insertions silently reordered
    # widgets added after the first raw insertion. No separate "Output
    # Directory" section any more - select_outdir_btn now lives inside
    # "Output Settings".
    layout = widget._content_native.layout()
    titles = []
    for i in range(layout.count()):
        wdg = layout.itemAt(i).widget()
        toggle = getattr(wdg, "toggleButton", None)
        titles.append(toggle().text() if toggle else type(wdg).__name__)

    assert titles == [
        "QLabel",
        "QWidget",
        "Physical spacing",
        "Extraction layers",
        "Cleanup",
        "Advanced features",
        "Output settings",
        "Recipe",
        "QPushButton",
    ]
    assert isinstance(layout.itemAt(layout.count() - 1).widget(), QPushButton)


def test_image_selector_label_preserved(widget):
    from qtpy.QtWidgets import QLabel

    labels = [lbl.text() for lbl in widget._content_native.findChildren(QLabel)]
    assert "Input segmentation (labels layer)" in labels


# -- logo ---------------------------------------------------------------


def test_logo_is_first_widget_in_layout(widget):
    layout = widget._content_native.layout()
    from qtpy.QtWidgets import QLabel as _QLabel

    assert isinstance(layout.itemAt(0).widget(), _QLabel)


def test_logo_pixmap_loaded_and_scaled_small(widget):
    from napari_maskel._napari import _LOGO_DISPLAY_HEIGHT

    layout = widget._content_native.layout()
    logo_label = layout.itemAt(0).widget()
    pixmap = logo_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.height() == _LOGO_DISPLAY_HEIGHT


def test_logo_is_left_aligned(widget):
    from qtpy.QtCore import Qt

    layout = widget._content_native.layout()
    logo_label = layout.itemAt(0).widget()
    assert logo_label.alignment() & Qt.AlignLeft


def test_show_preprocessed_ordered_before_prune_spurs_and_junction_cleanup(widget):
    names = [w.name for w in widget._extraction_gui]
    assert names.index("show_preprocessed") < names.index("prune_spurs")
    assert names.index("show_preprocessed") < names.index("junction_cleanup")


def test_show_preprocessed_label(widget):
    assert widget.show_preprocessed_widget.label == "Show preprocessed mask"


def test_relabeled_widgets(widget):
    assert widget.fill_holes_widget.label == "Fill holes in mask"
    assert widget.junction_cleanup_widget.label == "Skeleton junction cleanup"
    assert widget.prune_spurs_widget.label == "Prune skeleton spurs"
    assert widget.load_btn.text == "Load recipe"
    assert widget.save_btn.text == "Save recipe"
    assert widget.analyze_btn.text == "Analyze mask"


def test_prune_spurs_toggle_still_enables_dependents(widget):
    """Reordering must not have disturbed the .changed.connect wiring."""
    assert widget.min_spur_length_widget.enabled is False
    assert widget.spur_iterations_widget.enabled is False

    widget.prune_spurs_widget.value = True
    assert widget.min_spur_length_widget.enabled is True
    assert widget.spur_iterations_widget.enabled is True

    widget.prune_spurs_widget.value = False
    assert widget.min_spur_length_widget.enabled is False
    assert widget.spur_iterations_widget.enabled is False


def test_fill_holes_toggle_still_enables_show_preprocessed(widget):
    assert widget.show_preprocessed_widget.enabled is False

    widget.fill_holes_widget.value = True
    assert widget.show_preprocessed_widget.enabled is True

    widget.fill_holes_widget.value = False
    assert widget.show_preprocessed_widget.enabled is False


# -- output directory warning ------------------------------------------------


def test_output_dir_warning_visible_by_default(widget):
    # write_skeleton_npy and write_summary_csv default to True, and no
    # output directory is selected yet.
    assert widget.output_dir_warning.visible is True


def test_output_dir_warning_hidden_when_no_write_option_active(widget):
    for w in widget._write_option_widgets:
        w.value = False
    assert widget.output_dir_warning.visible is False


def test_output_dir_warning_hidden_once_output_dir_selected(widget):
    assert widget.output_dir_warning.visible is True
    widget._output_dir = "/tmp/fake"
    widget._update_output_dir_controls()
    assert widget.output_dir_warning.visible is False


def test_output_dir_warning_reappears_if_output_dir_cleared(widget):
    widget._output_dir = "/tmp/fake"
    widget._update_output_dir_controls()
    assert widget.output_dir_warning.visible is False

    widget._output_dir = None
    widget._update_output_dir_controls()
    assert widget.output_dir_warning.visible is True


def test_output_dir_warning_reacts_to_any_single_write_option(widget):
    for w in widget._write_option_widgets:
        w.value = False
    assert widget.output_dir_warning.visible is False

    widget.write_graphml_widget.value = True
    assert widget.output_dir_warning.visible is True


def test_output_dir_button_disabled_when_no_write_option_active(widget):
    assert widget.select_outdir_btn.enabled is True  # npy+summary csv default on

    for w in widget._write_option_widgets:
        w.value = False
    assert widget.select_outdir_btn.enabled is False

    widget.write_graphml_widget.value = True
    assert widget.select_outdir_btn.enabled is True


def test_output_dir_button_lives_in_output_settings_group(widget):
    from qtpy.QtWidgets import QPushButton as QPB

    output_settings_native = widget._output_collapsible
    buttons = [
        b.text()
        for b in output_settings_native.findChildren(QPB)
        if b.text() == "Select output directory..."
    ]
    assert buttons == ["Select output directory..."]


def test_selecting_output_dir_preserves_path_after_disable_and_reenable(widget):
    widget._output_dir = "/tmp/fake"
    widget.select_outdir_btn.text = "/tmp/fake"

    for w in widget._write_option_widgets:
        w.value = False
    assert widget.select_outdir_btn.enabled is False
    assert widget._output_dir == "/tmp/fake"

    widget.write_graphml_widget.value = True
    assert widget.select_outdir_btn.enabled is True
    assert widget._output_dir == "/tmp/fake"
    assert widget.output_dir_warning.visible is False


# -- networkx graph export ---------------------------------------------------


def test_write_networkx_graph_widget_label(widget):
    assert widget.write_networkx_graph_widget.label == "Write networkx graph (.pkl)"


def test_write_networkx_graph_is_a_write_option(widget):
    assert widget.write_networkx_graph_widget in widget._write_option_widgets


# -- checkbox text actually rendered on the native Qt widget -----------------
#
# CheckBox is the one magicgui widget type that displays its own caption
# directly on the native Qt control via a separate `.text` attribute, rather
# than through the external QLabel that `.label` normally controls - so
# `widget.label == "..."` alone (as asserted above) can pass even when the
# *rendered* checkbox still shows old/auto-derived text. These checks catch
# that class of bug by reading `.native.text()`, what a user actually sees.
_CHECKBOX_LABELS = {
    "extract_branches_widget": "Extract branch features",
    "extract_branch_text_widget": "Add branch labels",
    "extract_summary_widget": "Extract object-level features",
    "extract_nodes_widget": "Extract node features",
    "fill_holes_widget": "Fill holes in mask",
    "show_preprocessed_widget": "Show preprocessed mask",
    "junction_cleanup_widget": "Skeleton junction cleanup",
    "prune_spurs_widget": "Prune skeleton spurs",
    "write_skeleton_npy_widget": "Write skeleton (.npy)",
    "write_skeleton_png_widget": "Write skeleton (.png)",
    "write_summary_csv_widget": "Write summary csv",
    "write_radius_widget": "Write radius matrix (.npy)",
    "write_branch_csv_widget": "Write branch csv",
    "write_node_csv_widget": "Write node csv",
    "write_graphml_widget": "Write graph (.graphml)",
    "write_networkx_graph_widget": "Write networkx graph (.pkl)",
}


@pytest.mark.parametrize("attr_name, expected", _CHECKBOX_LABELS.items())
def test_checkbox_native_text_matches_intended_label(widget, attr_name, expected):
    checkbox = getattr(widget, attr_name)
    assert checkbox.native.text() == expected
    assert checkbox.label == expected


def test_write_networkx_graph_round_trips_through_pipeline_config(widget):
    widget.write_networkx_graph_widget.value = True
    config = widget._get_current_pipeline_config()
    assert config.output.write_networkx_graph is True

    fresh = MaskAnalysisWidget(MagicMock())
    try:
        fresh._set_pipeline_config(config)
        assert fresh.write_networkx_graph_widget.value is True
    finally:
        fresh.close()


# -- image shape label -------------------------------------------------------


def test_image_shape_label_empty_with_no_image_selected(widget):
    assert widget.image_shape_label.value == ""


def test_image_shape_label_shows_shape_on_image_change(widget):
    layer = _mock_layer((12, 34, 56))
    _select_image(widget, layer)
    widget.image_widget.changed.emit(layer)
    assert "(12, 34, 56)" in widget.image_shape_label.value


def test_image_dependent_widgets_sync_for_a_pre_selected_image(widget):
    # image_widget can already hold a layer at construction time (magicgui
    # auto-selects one of its choices) without any .changed signal ever
    # having fired for it - _sync_image_dependent_widgets is what backfills
    # the shape label / default spacing / spacing warning for that case,
    # so call it directly rather than through _select_image (which fires
    # .changed and would mask a regression here).
    layer = _mock_layer((12, 34, 56))
    widget.image_widget.choices = (layer,)
    widget.image_widget.value = layer
    widget.image_shape_label.value = ""
    widget.spacing_widget.value = ""

    widget._sync_image_dependent_widgets()

    assert "(12, 34, 56)" in widget.image_shape_label.value
    assert widget.spacing_widget.value == "1,1,1"


def test_image_shape_label_wraps_instead_of_overflowing(widget):
    # Word wrap alone isn't enough - magicgui's Label backend hard-codes
    # QSizePolicy.Fixed on both axes, so the label is always sized to its
    # own unconstrained (one-line) sizeHint and never shrunk by the layout
    # no matter what setWordWrap() says. The horizontal policy has to be
    # overridden too (see _prepare_wrapping_label) or the widget's real
    # bounding box still silently grows past the dock panel instead of
    # wrapping, and the overflow just runs off-screen.
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy

    native = widget.image_shape_label.native
    assert native.wordWrap() is True
    assert native.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert native.alignment() & Qt.AlignLeft


# -- dependent widgets auto-uncheck ------------------------------------------


@pytest.mark.parametrize(
    "parent_name,dependent_name",
    [
        ("extract_branches_widget", "extract_branch_text_widget"),
        ("extract_branches_widget", "write_branch_csv_widget"),
        ("include_mask_radius_widget", "write_radius_widget"),
        ("extract_summary_widget", "write_summary_csv_widget"),
        ("extract_nodes_widget", "write_node_csv_widget"),
    ],
)
def test_dependent_checkbox_auto_unchecks_when_parent_turns_off(
    widget, parent_name, dependent_name
):
    parent = getattr(widget, parent_name)
    dependent = getattr(widget, dependent_name)

    parent.value = True
    dependent.value = True
    assert dependent.enabled is True

    parent.value = False
    assert dependent.enabled is False
    assert dependent.value is False


def test_show_preprocessed_auto_unchecks_when_fill_holes_and_closing_off(widget):
    widget.fill_holes_widget.value = True
    widget.show_preprocessed_widget.value = True
    assert widget.show_preprocessed_widget.enabled is True

    widget.fill_holes_widget.value = False
    assert widget.show_preprocessed_widget.enabled is False
    assert widget.show_preprocessed_widget.value is False


def test_show_preprocessed_stays_enabled_via_closing_iterations_alone(widget):
    widget.closing_iterations_widget.value = 2
    widget.show_preprocessed_widget.value = True

    widget.fill_holes_widget.value = True
    widget.fill_holes_widget.value = False
    assert widget.show_preprocessed_widget.enabled is True
    assert widget.show_preprocessed_widget.value is True


def test_loading_config_with_inconsistent_state_is_reconciled(widget):
    config = PipelineConfig(
        extraction=ExtractionConfig(mask_radius=False),
        output=OutputConfig(write_radius=True),
    )
    widget._set_pipeline_config(config)
    assert widget.write_radius_widget.value is False
    assert widget.write_radius_widget.enabled is False


# -- spacing field ----------------------------------------------------------


def test_spacing_field_defaults_to_empty_string(widget):
    assert widget.spacing_widget.value == ""


def test_spacing_warning_hidden_by_default(widget):
    assert widget.spacing_warning.visible is False


def test_spacing_warning_wraps_instead_of_overflowing(widget):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy

    native = widget.spacing_warning.native
    assert native.wordWrap() is True
    assert native.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert native.alignment() & Qt.AlignLeft


def test_get_current_spacing_none_with_no_image_selected(widget):
    assert widget.image_widget.value is None
    assert widget._get_current_spacing() is None


def test_get_current_spacing_parses_valid_value(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "2.0,0.5"
    assert widget._get_current_spacing() == (2.0, 0.5)


def test_get_current_spacing_falls_back_on_dimension_mismatch(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "1.0,1.0,1.0"
    assert widget._get_current_spacing() is None


def test_get_current_spacing_falls_back_on_unparsable_text(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "not,numbers"
    assert widget._get_current_spacing() is None


def test_spacing_warning_visible_on_invalid_input(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "1.0,1.0,1.0"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    assert widget.spacing_warning.visible is True


def test_spacing_warning_hidden_on_valid_input(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "1.0,1.0,1.0"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    assert widget.spacing_warning.visible is True

    widget.spacing_widget.value = "1.0,1.0"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    assert widget.spacing_warning.visible is False


def test_pipeline_config_round_trip_preserves_spacing(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "2.0,0.5"

    config = widget._get_current_pipeline_config()
    assert config.extraction.spacing == (2.0, 0.5)

    fresh = MaskAnalysisWidget(MagicMock())
    try:
        fresh._set_pipeline_config(config)
        assert fresh.spacing_widget.value == "2.0,0.5"
    finally:
        fresh.close()


def test_set_pipeline_config_with_none_spacing_clears_field(widget):
    widget.spacing_widget.value = "2.0,0.5"
    config = PipelineConfig(
        extraction=ExtractionConfig(spacing=None), output=OutputConfig()
    )
    widget._set_pipeline_config(config)
    assert widget.spacing_widget.value == ""


def test_invalid_spacing_not_included_in_pipeline_config(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "1.0,1.0,1.0"  # wrong ndim for a 2D image
    config = widget._get_current_pipeline_config()
    assert config.extraction.spacing is None


def test_spacing_field_defaults_from_layer_scale_on_image_change(widget):
    layer = _mock_layer((10, 10, 10))
    layer.scale = (2.0, 0.5, 0.5)
    _select_image(widget, layer)
    widget.image_widget.changed.emit(layer)
    assert widget.spacing_widget.value == "2,0.5,0.5"


# -- fractal-dimension anisotropic-spacing warning ---------------------------


def test_fractal_anisotropic_warning_hidden_by_default(widget):
    assert widget.fractal_anisotropic_warning.visible is False


def test_fractal_anisotropic_warning_shown_when_fractal_and_anisotropic(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "2.0,0.5"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    widget.include_fractal_widget.value = True
    assert widget.fractal_anisotropic_warning.visible is True


def test_fractal_anisotropic_warning_hidden_for_isotropic_spacing(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "2.0,2.0"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    widget.include_fractal_widget.value = True
    assert widget.fractal_anisotropic_warning.visible is False


def test_fractal_anisotropic_warning_hidden_when_fractal_disabled(widget):
    _select_image(widget, _mock_layer((10, 10)))
    widget.spacing_widget.value = "2.0,0.5"
    widget.spacing_widget.changed.emit(widget.spacing_widget.value)
    widget.include_fractal_widget.value = True
    assert widget.fractal_anisotropic_warning.visible is True

    widget.include_fractal_widget.value = False
    assert widget.fractal_anisotropic_warning.visible is False


# -- _on_analyze --------------------------------------------------------------


class TestOnAnalyze:
    def test_no_image_selected_does_nothing(self, widget):
        assert widget.image_widget.value is None  # no layer selected yet
        widget._on_analyze()
        widget.viewer.add_layer.assert_not_called()

    def test_happy_path_adds_layers(self, widget):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)
        widget.extract_branches_widget.value = True

        widget._on_analyze()

        assert widget.viewer.add_layer.called

    def test_show_preprocessed_adds_extra_layer(self, widget):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)
        widget.fill_holes_widget.value = True
        widget.show_preprocessed_widget.value = True

        widget._on_analyze()

        added_names = [
            call.args[0].name for call in widget.viewer.add_layer.call_args_list
        ]
        assert f"{layer.name}_preprocessed" in added_names

    def test_layer_create_failure_for_one_layer_does_not_abort_others(
        self, widget, monkeypatch
    ):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)
        widget.extract_branches_widget.value = True

        real_create = napari_maskel_module.Layer.create

        def flaky_create(data, meta, layer_type):
            if meta.get("name", "").endswith("_branches"):
                raise ValueError("boom")
            return real_create(data, meta, layer_type)

        monkeypatch.setattr(
            napari_maskel_module.Layer, "create", staticmethod(flaky_create)
        )
        infos = []
        monkeypatch.setattr(
            napari_maskel_module, "show_info", lambda m: infos.append(m)
        )

        widget._on_analyze()  # must not raise

        assert any("Failed to add layer" in m and "_branches" in m for m in infos)
        assert widget.viewer.add_layer.called  # other layers still got through

    def test_output_dir_set_writes_files(self, widget, tmp_path):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)
        widget._output_dir = tmp_path

        widget._on_analyze()

        out_dir = tmp_path / layer.name
        assert out_dir.exists()
        assert (out_dir / f"{layer.name}_skeleton.npy").exists()

    def test_output_write_failure_reports_distinct_message(
        self, widget, tmp_path, monkeypatch
    ):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)
        widget._output_dir = tmp_path

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(napari_maskel_module, "save_analysis_outputs", boom)
        errors = []
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_analyze()  # must not raise

        assert len(errors) == 1
        assert "saving results failed" in errors[0]
        assert "Analysis failed" not in errors[0]

    def test_analysis_failure_reports_analysis_failed(self, widget, monkeypatch):
        layer = _real_image_layer(_small_cross_mask())
        _select_image(widget, layer)

        def boom(mask, config):
            raise ValueError("bad mask")

        monkeypatch.setattr(napari_maskel_module, "analyze_segmentation_mask", boom)
        errors = []
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_analyze()

        assert len(errors) == 1
        assert "Analysis failed" in errors[0]


# -- config load/save/output-dir dialogs --------------------------------------


class TestOnLoadConfig:
    def test_cancel_is_a_noop(self, widget, monkeypatch):
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        infos, errors = [], []
        monkeypatch.setattr(
            napari_maskel_module, "show_info", lambda m: infos.append(m)
        )
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_load_config()

        assert infos == []
        assert errors == []

    def test_success_updates_widget_state(self, widget, monkeypatch, tmp_path):
        config = PipelineConfig(
            extraction=ExtractionConfig(branches=True), output=OutputConfig()
        )
        path = tmp_path / "recipe.json"
        save_pipeline_config(config, path)

        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: (str(path), ""),
        )
        infos = []
        monkeypatch.setattr(
            napari_maskel_module, "show_info", lambda m: infos.append(m)
        )

        widget._on_load_config()

        assert widget.extract_branches_widget.value is True
        assert infos == ["Configuration loaded"]

    def test_malformed_json_reports_error_instead_of_crashing(
        self, widget, monkeypatch, tmp_path
    ):
        # "extraction" present but not an object - PipelineConfig.from_dict
        # raises TypeError for this (not ValueError), which is exactly what
        # the widened except clause now catches instead of crashing.
        path = tmp_path / "bad.json"
        path.write_text('{"schema_version": 6, "extraction": "oops", "output": {}}')

        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: (str(path), ""),
        )
        errors = []
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_load_config()  # must not raise

        assert len(errors) == 1
        assert "Failed to load config" in errors[0]

    def test_nonexistent_file_reports_error(self, widget, monkeypatch, tmp_path):
        path = tmp_path / "missing.json"
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: (str(path), ""),
        )
        errors = []
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_load_config()

        assert len(errors) == 1
        assert "Failed to load config" in errors[0]


class TestOnSaveConfig:
    def test_cancel_is_a_noop(self, widget, monkeypatch):
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getSaveFileName",
            lambda *a, **k: ("", ""),
        )
        infos, errors = [], []
        monkeypatch.setattr(
            napari_maskel_module, "show_info", lambda m: infos.append(m)
        )
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_save_config()

        assert infos == []
        assert errors == []

    def test_success_writes_real_file(self, widget, monkeypatch, tmp_path):
        widget.extract_branches_widget.value = True
        path = tmp_path / "out.json"

        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getSaveFileName",
            lambda *a, **k: (str(path), ""),
        )
        infos = []
        monkeypatch.setattr(
            napari_maskel_module, "show_info", lambda m: infos.append(m)
        )

        widget._on_save_config()

        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["extraction"]["branches"] is True
        assert infos == [f"Configuration saved to {path}"]

    def test_failure_reports_error(self, widget, monkeypatch, tmp_path):
        path = tmp_path / "out.json"
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getSaveFileName",
            lambda *a, **k: (str(path), ""),
        )

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(napari_maskel_module, "save_pipeline_config", boom)
        errors = []
        monkeypatch.setattr(
            napari_maskel_module, "show_error", lambda m: errors.append(m)
        )

        widget._on_save_config()

        assert len(errors) == 1
        assert "Failed to save config" in errors[0]


class TestOnSelectOutputDir:
    def test_cancel_leaves_output_dir_unset(self, widget, monkeypatch):
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog, "getExistingDirectory", lambda *a, **k: ""
        )
        widget._on_select_output_dir()
        assert widget._output_dir is None

    def test_success_sets_output_dir_and_updates_button(
        self, widget, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            napari_maskel_module.QFileDialog,
            "getExistingDirectory",
            lambda *a, **k: str(tmp_path),
        )
        widget._on_select_output_dir()
        assert widget._output_dir == tmp_path
        assert widget.select_outdir_btn.text == str(tmp_path)


# -- _recolor_branch_layers ----------------------------------------------------


class TestRecolorBranchLayers:
    @staticmethod
    def _branch_layer(name, tortuosity_values):
        data = [np.array([[0, 0], [1, i + 1]]) for i in range(len(tortuosity_values))]
        return Shapes(
            data=data,
            shape_type="line",
            properties={"tortuosity": tortuosity_values},
            name=name,
        )

    def test_non_matching_layer_is_skipped_without_error(self, widget):
        layer = self._branch_layer("something_else", [1.0, 2.0])
        widget.viewer.layers = [layer]
        widget.branch_color_widget.value = "tortuosity"
        widget._recolor_branch_layers()  # must not raise

    def test_non_shapes_layer_is_skipped_without_error(self, widget):
        # Distinct from the name-mismatch case above: exercises the
        # isinstance(layer, Shapes) half of the guard's `or`, not the
        # name.endswith("_branches") half - a non-Shapes layer (e.g. a
        # Labels layer someone happened to name "..._branches") must be
        # skipped too, not just one with the wrong Shapes.
        layer = MagicMock()
        layer.name = "obj_branches"
        widget.viewer.layers = [layer]
        widget.branch_color_widget.value = "tortuosity"
        widget._recolor_branch_layers()  # must not raise

    def test_numeric_property_with_spread_uses_colormap(self, widget):
        layer = self._branch_layer("img_branches", [1.0, 2.0, 5.0])
        widget.viewer.layers = [layer]
        widget.branch_color_widget.value = "tortuosity"

        widget._recolor_branch_layers()

        assert layer.edge_color_mode == "colormap"
        assert layer.edge_colormap.name == "turbo"
        assert layer.edge_contrast_limits == (1.0, 5.0)

    def test_constant_property_falls_back_to_flat_color(self, widget):
        layer = self._branch_layer("img_branches", [3.0, 3.0, 3.0])
        widget.viewer.layers = [layer]
        widget.branch_color_widget.value = "tortuosity"

        widget._recolor_branch_layers()

        assert layer.edge_color_mode == "direct"

    def test_missing_property_falls_back_to_flat_color(self, widget):
        layer = self._branch_layer("img_branches", [1.0, 2.0])
        widget.viewer.layers = [layer]
        widget.branch_color_widget.value = "mean_radius"  # not on this layer

        widget._recolor_branch_layers()

        assert layer.edge_color_mode == "direct"
