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
        "_outdir_collapsible",
        "_config_collapsible",
    ):
        assert getattr(widget, name).isExpanded() is True


def test_group_order_matches_original_layout(widget):
    # image selector, then every group in its original relative order, then
    # the Analyze button last. Regression test for a bug where mixing
    # self.append() with raw layout insertions silently reordered widgets
    # added after the first raw insertion.
    layout = widget._content_native.layout()
    titles = []
    for i in range(layout.count()):
        wdg = layout.itemAt(i).widget()
        toggle = getattr(wdg, "toggleButton", None)
        titles.append(toggle().text() if toggle else type(wdg).__name__)

    assert titles == [
        "QWidget",
        "Physical Spacing",
        "Extraction Layers",
        "Cleanup",
        "Advanced Features",
        "Output Settings",
        "Output Directory",
        "Configuration",
        "QPushButton",
    ]
    assert isinstance(layout.itemAt(layout.count() - 1).widget(), QPushButton)


def test_image_selector_label_preserved(widget):
    from qtpy.QtWidgets import QLabel

    labels = [lbl.text() for lbl in widget._content_native.findChildren(QLabel)]
    assert "Input segmentation (labels layer)" in labels


def test_show_preprocessed_ordered_before_prune_spurs(widget):
    names = [w.name for w in widget._extraction_gui]
    assert names.index("show_preprocessed") < names.index("prune_spurs")


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
