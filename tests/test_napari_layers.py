import numpy as np
import pytest
from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.pipeline import analyze_segmentation_mask

from napari_maskel.napari_layers import extract_skeleton_layers


def _cross_mask(size: int = 32) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    img[size // 2, :] = 1
    img[:, size // 2] = 1
    return img


def _two_object_mask() -> np.ndarray:
    """40x40 canvas with two disjoint 12px crosses, labeled 5 and 9."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5, 2:14] = 5
    mask[2:14, 8] = 5
    mask[25, 22:34] = 9
    mask[22:34, 28] = 9
    return mask


def _analyze(image, **extraction_kwargs):
    config = PipelineConfig(
        extraction=ExtractionConfig(**extraction_kwargs), output=OutputConfig()
    )
    return analyze_segmentation_mask(image, config)


class TestExtractSkeletonLayers:
    @pytest.fixture
    def result(self):
        return _analyze(_cross_mask(), branches=True, summary=True)

    def test_returns_layer_tuples(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=True, summary=True)
        )
        assert len(layers) > 0
        for layer in layers:
            assert isinstance(layer, tuple) and len(layer) == 3
            assert isinstance(layer[1], dict)
            assert isinstance(layer[2], str)

    def test_includes_branch_layer(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=True, summary=True)
        )
        assert "shapes" in [layer[2] for layer in layers]

    def test_includes_summary_layer(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=True, summary=True)
        )
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert any("_summary" in name for name in layer_names)

    def test_includes_two_point_layers_with_branch_text(self, result):
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(branches=True, branch_text=True, summary=True),
        )
        layer_types = [layer[2] for layer in layers]
        assert layer_types.count("points") == 2

    def test_branches_disabled(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=False)
        )
        assert "shapes" not in [layer[2] for layer in layers]

    def test_branch_text_disabled(self, result):
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(branches=True, branch_text=False, summary=True),
        )
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert any("_summary" in name for name in layer_names)
        assert not any("branch_text" in name for name in layer_names)

    def test_summary_disabled(self, result):
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(branches=True, branch_text=True, summary=False),
        )
        layer_types = [layer[2] for layer in layers]
        assert "points" in layer_types
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert not any("summary" in name for name in layer_names)

    def test_summary_layer_has_properties(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(summary=True)
        )
        summary_layer = [
            layer for layer in layers if layer[1].get("name", "").endswith("_summary")
        ]
        assert len(summary_layer) == 1
        assert "properties" in summary_layer[0][1]

    def test_with_radius_matrix(self):
        result = _analyze(_cross_mask(), summary=True, vessel_radius=True)
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(summary=True, vessel_radius=True),
        )
        assert "image" in [layer[2] for layer in layers]

    def test_layer_names_include_base_name(self, result):
        layers = extract_skeleton_layers(
            result, "my_vessels", config=ExtractionConfig(branches=True, summary=True)
        )
        assert len(layers) > 0
        for layer in layers:
            assert "my_vessels" in layer[1].get("name", "")

    def test_branch_layer_has_properties(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=True, summary=True)
        )
        branch_layer = next(layer for layer in layers if layer[2] == "shapes")
        assert len(branch_layer[1]["properties"]) > 0

    def test_empty_skeleton_produces_no_layers(self):
        result = _analyze(
            np.zeros((10, 10), dtype=np.uint8), branches=True, summary=True
        )
        layers = extract_skeleton_layers(
            result, "empty", config=ExtractionConfig(branches=True, summary=True)
        )
        assert layers == []

    def test_branch_text_layer_has_text_config(self, result):
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(branches=True, branch_text=True, summary=True),
        )
        text_layers = [
            layer
            for layer in layers
            if layer[2] == "points"
            and layer[1].get("name", "").endswith("_branch_text")
        ]
        assert len(text_layers) == 1
        text_config = text_layers[0][1]["text"]
        assert "string" in text_config
        assert "size" in text_config


class TestMultiObjectLayers:
    @pytest.fixture
    def result(self):
        return _analyze(_two_object_mask(), branches=True, nodes=True, summary=True)

    def test_branch_layer_has_object_id_property(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(branches=True, summary=True)
        )
        branch_layer = next(layer for layer in layers if layer[2] == "shapes")
        props = branch_layer[1]["properties"]
        assert set(props["object_id"]) == {5, 9}

    def test_branch_layer_colorable_by_object_id(self, result):
        layers = extract_skeleton_layers(
            result,
            "test",
            config=ExtractionConfig(
                branches=True, summary=True, branch_color_property="object_id"
            ),
        )
        branch_layer = next(layer for layer in layers if layer[2] == "shapes")
        assert branch_layer[1]["edge_color"] == "object_id"
        assert branch_layer[1]["edge_colormap"] == "turbo"

    def test_summary_layer_has_one_point_per_object(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(summary=True)
        )
        summary_layer = next(
            layer for layer in layers if layer[1].get("name", "").endswith("_summary")
        )
        assert len(summary_layer[0]) == 2
        assert set(summary_layer[1]["properties"]["object_id"]) == {5, 9}

    def test_node_layer_has_object_id_property(self, result):
        layers = extract_skeleton_layers(
            result, "test", config=ExtractionConfig(nodes=True)
        )
        node_layer = next(
            layer for layer in layers if layer[1].get("name", "").endswith("_nodes")
        )
        assert set(node_layer[1]["properties"]["object_id"]) == {5, 9}
