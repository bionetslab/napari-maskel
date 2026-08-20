"""Smoke tests for MaskAnalysisWidget construction and spacing-field wiring.

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
