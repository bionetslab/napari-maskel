"""Smoke tests for MaskAnalysisWidget construction, layout, and spacing-field wiring.

These run headless (QT_QPA_PLATFORM=offscreen) and use a plain mock in place
of a real napari.Viewer, since MaskAnalysisWidget.__init__ never calls
anything on the viewer during construction - it's only used later, in
_recolor_branch_layers/_on_analyze. That sidesteps needing napari's
make_napari_viewer fixture, which itself needs pytest-qt (not a project
dependency here).
"""

import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from qtpy.QtWidgets import QPushButton, QScrollArea

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
    # image selector, then every group in its original relative order, then
    # the Analyze button last. Regression test for a bug where mixing
    # self.append() with raw layout insertions silently reordered widgets
    # added after the first raw insertion. No separate "Output Directory"
    # section any more - select_outdir_btn now lives inside "Output Settings".
    layout = widget._content_native.layout()
    titles = []
    for i in range(layout.count()):
        wdg = layout.itemAt(i).widget()
        toggle = getattr(wdg, "toggleButton", None)
        titles.append(toggle().text() if toggle else type(wdg).__name__)

    assert titles == [
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
    "extract_branches_widget": "Extract branches",
    "extract_branch_text_widget": "Add branch labels",
    "extract_summary_widget": "Extract summary statistics",
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
