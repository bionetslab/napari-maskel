import numpy as np
import pytest
from maskel.config import ExtractionConfig
from maskel.features import build_vessel_graph, compute_radii, extract_vessel_features
from skan import summarize

from napari_maskel.napari_layers import extract_skeleton_layers


def _make_2d_cross_skeleton() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[16, :] = 1
    img[:, 16] = 1
    return img


def _features_for(skeleton, graph, branch_data):
    return extract_vessel_features(
        skeleton, graph, branch_data, binary=skeleton, include_fractal=False
    )


class TestExtractSkeletonLayers:
    @pytest.fixture
    def skeleton(self):
        return _make_2d_cross_skeleton()

    @pytest.fixture
    def graph(self, skeleton):
        return build_vessel_graph(skeleton)

    @pytest.fixture
    def branch_data(self, graph):
        return summarize(graph, separator="-")

    @pytest.fixture
    def features(self, skeleton, graph, branch_data):
        return _features_for(skeleton, graph, branch_data)

    def test_with_config_returns_layers(self, skeleton, graph, branch_data, features):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, summary=True),
            features=features,
        )
        assert len(layers) > 0
        for layer in layers:
            assert isinstance(layer, tuple) and len(layer) == 3
            assert isinstance(layer[1], dict)
            assert isinstance(layer[2], str)

    def test_with_config_includes_branch_layer(
        self, skeleton, graph, branch_data, features
    ):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, summary=True),
            features=features,
        )
        layer_types = [layer[2] for layer in layers]
        assert "shapes" in layer_types

    def test_with_config_includes_summary_layer(
        self, skeleton, graph, branch_data, features
    ):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, summary=True),
            features=features,
        )
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert any("_summary" in name for name in layer_names)

    def test_with_config_includes_two_point_layers(
        self, skeleton, graph, branch_data, features
    ):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, branch_text=True, summary=True),
            features=features,
        )
        layer_types = [layer[2] for layer in layers]
        num_points = layer_types.count("points")
        assert num_points == 2

    def test_branches_disabled(self, skeleton, graph, branch_data, features):
        config = ExtractionConfig(branches=False)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=config,
            features=features,
        )
        layer_types = [layer[2] for layer in layers]
        assert "shapes" not in layer_types

    def test_branch_text_disabled(self, skeleton, graph, branch_data, features):
        config = ExtractionConfig(branches=True, branch_text=False, summary=True)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=config,
            features=features,
        )
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert any("_summary" in name for name in layer_names)
        assert not any("branch_text" in name for name in layer_names)

    def test_summary_disabled(self, skeleton, graph, branch_data):
        config = ExtractionConfig(branches=True, branch_text=True, summary=False)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=config,
        )
        layer_types = [layer[2] for layer in layers]
        assert "points" in layer_types
        layer_names = [layer[1].get("name", "") for layer in layers]
        assert not any("summary" in name for name in layer_names)

    def test_with_features_passed(self, skeleton, graph, branch_data, features):
        config = ExtractionConfig(summary=True)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=config,
            features=features,
        )
        summary_layer = [
            layer for layer in layers if layer[1].get("name", "").endswith("_summary")
        ]
        assert len(summary_layer) == 1
        assert "properties" in summary_layer[0][1]

    def test_with_radius_matrix(self, skeleton, graph, branch_data, features):
        radius_matrix, _ = compute_radii(skeleton, skeleton)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            features=features,
            radius_matrix=radius_matrix,
        )
        layer_types = [layer[2] for layer in layers]
        assert "image" in layer_types

    def test_with_empty_radius_matrix_skips_layer(
        self, skeleton, graph, branch_data, features
    ):
        radius_matrix = np.zeros_like(skeleton, dtype=np.float64)
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            features=features,
            radius_matrix=radius_matrix,
        )
        layer_types = [layer[2] for layer in layers]
        assert "image" not in layer_types

    def test_layer_names_include_base_name(
        self, skeleton, graph, branch_data, features
    ):
        base_name = "my_vessels"
        layers = extract_skeleton_layers(
            skeleton,
            base_name,
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, summary=True),
            features=features,
        )
        assert len(layers) > 0
        for layer in layers:
            assert base_name in layer[1].get("name", "")

    def test_branch_layer_has_properties(self, skeleton, graph, branch_data, features):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, summary=True),
            features=features,
        )
        branch_layer = next(layer for layer in layers if layer[2] == "shapes")
        props = branch_layer[1]["properties"]
        assert len(props) > 0

    def test_empty_branch_data_skips_shapes_layer(self, skeleton, graph, features):
        data = summarize(graph, separator="-").iloc[0:0]
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            data,
            config=ExtractionConfig(branches=True, branch_text=False),
            features=features,
        )
        layer_types = [layer[2] for layer in layers]
        assert "shapes" not in layer_types

    def test_empty_skeleton(self):
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        seeds = _make_2d_cross_skeleton()
        graph = build_vessel_graph(seeds)
        branch_data = summarize(graph, separator="-")
        features = _features_for(seeds, graph, branch_data)
        layers = extract_skeleton_layers(
            skeleton,
            "empty",
            graph,
            branch_data,
            config=ExtractionConfig(summary=True),
            features=features,
        )
        summary = [layer for layer in layers if "_summary" in layer[1].get("name", "")]
        assert len(summary) == 1

    def test_branch_text_layer_has_text_config(
        self, skeleton, graph, branch_data, features
    ):
        layers = extract_skeleton_layers(
            skeleton,
            "test",
            graph,
            branch_data,
            config=ExtractionConfig(branches=True, branch_text=True, summary=True),
            features=features,
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

    def test_summary_with_none_features_crashes(self, skeleton, graph, branch_data):
        config = ExtractionConfig(summary=True)
        with pytest.raises(ValueError, match="features is required"):
            extract_skeleton_layers(
                skeleton,
                "test",
                graph,
                branch_data,
                config=config,
                features=None,
            )
