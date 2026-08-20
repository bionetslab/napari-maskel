"""Tests for the 3D + PNG-skeleton-export warning label.

Run headless (QT_QPA_PLATFORM=offscreen) like test_napari_widget.py. The
widget/warning label must be shown via ``widget.show()`` before checking
``.visible`` - magicgui's ``Widget.visible`` proxies straight to Qt's
``isVisible()``, which is only meaningful once the ancestor chain has
actually been shown.
"""

import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from napari_maskel._napari import MaskAnalysisWidget


@pytest.fixture
def widget():
    w = MaskAnalysisWidget(MagicMock())
    w.show()
    yield w
    w.close()


def _mock_image(ndim: int) -> MagicMock:
    img = MagicMock()
    img.data = np.zeros((4,) * ndim)
    return img


def test_warning_hidden_by_default(widget):
    assert widget.write_skeleton_png_warning.visible is False


def test_warning_shown_for_3d_image_with_png_enabled(widget):
    img3d = _mock_image(3)
    widget.image_widget.choices = (img3d,)
    widget.image_widget.value = img3d

    widget.write_skeleton_png_widget.value = True
    assert widget.write_skeleton_png_warning.visible is True


def test_warning_hidden_for_2d_image_with_png_enabled(widget):
    img2d = _mock_image(2)
    widget.image_widget.choices = (img2d,)
    widget.image_widget.value = img2d

    widget.write_skeleton_png_widget.value = True
    assert widget.write_skeleton_png_warning.visible is False


def test_warning_hidden_when_png_disabled_even_for_3d_image(widget):
    img3d = _mock_image(3)
    widget.image_widget.choices = (img3d,)
    widget.image_widget.value = img3d
    widget.write_skeleton_png_widget.value = True
    assert widget.write_skeleton_png_warning.visible is True

    widget.write_skeleton_png_widget.value = False
    assert widget.write_skeleton_png_warning.visible is False


def test_warning_updates_when_switching_image_layers(widget):
    img3d = _mock_image(3)
    img2d = _mock_image(2)
    widget.image_widget.choices = (img3d, img2d)
    widget.image_widget.value = img3d
    widget.write_skeleton_png_widget.value = True
    assert widget.write_skeleton_png_warning.visible is True

    widget.image_widget.value = img2d
    assert widget.write_skeleton_png_warning.visible is False

    widget.image_widget.value = img3d
    assert widget.write_skeleton_png_warning.visible is True


def test_warning_hidden_when_no_image_selected(widget):
    # image_widget.value defaults to None with no layers in the (mocked)
    # viewer - the ndim check must guard against that rather than raising.
    assert widget.image_widget.value is None
    widget.write_skeleton_png_widget.value = True
    assert widget.write_skeleton_png_warning.visible is False
