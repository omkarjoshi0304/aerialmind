"""Tests for the CPU fallback accelerator provider.

Verifies:
1. CPUFallbackProvider satisfies the AcceleratorHAL protocol
2. Capabilities report correct values
3. Error handling for missing models and unknown model_ids
4. Load/infer roundtrip with a minimal ONNX model (requires onnxruntime)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerialmind.core.protocols import AcceleratorHAL
from aerialmind.hal.accelerator.cpu_provider import CPUFallbackProvider

# ---------------------------------------------------------------------------
# Tests that do NOT require onnxruntime
# ---------------------------------------------------------------------------


class TestCPUFallbackProviderBasic:
    @pytest.fixture()
    def provider(self) -> CPUFallbackProvider:
        return CPUFallbackProvider()

    def test_conforms_to_accelerator_hal_protocol(
        self, provider: CPUFallbackProvider,
    ) -> None:
        assert isinstance(provider, AcceleratorHAL)

    def test_get_capabilities_has_expected_keys(
        self, provider: CPUFallbackProvider,
    ) -> None:
        caps = provider.get_capabilities()
        assert "precision" in caps
        assert "max_batch" in caps
        assert "device_name" in caps

    def test_get_capabilities_reports_cpu(
        self, provider: CPUFallbackProvider,
    ) -> None:
        caps = provider.get_capabilities()
        assert caps["device_name"] == "cpu"

    def test_get_capabilities_fp32_only(
        self, provider: CPUFallbackProvider,
    ) -> None:
        caps = provider.get_capabilities()
        assert caps["precision"] == ["fp32"]

    def test_unload_nonexistent_model_raises(
        self, provider: CPUFallbackProvider,
    ) -> None:
        with pytest.raises(KeyError, match="Unknown model_id"):
            provider.unload_model("nonexistent")

    def test_infer_nonexistent_model_raises(
        self, provider: CPUFallbackProvider,
    ) -> None:
        with pytest.raises(KeyError, match="Unknown model_id"):
            provider.infer("nonexistent", {})


# ---------------------------------------------------------------------------
# Tests that require onnxruntime (skipped if not installed)
# ---------------------------------------------------------------------------


class TestCPUFallbackProviderWithOnnx:
    @pytest.fixture(autouse=True)
    def _require_onnxruntime(self) -> None:
        pytest.importorskip("onnxruntime")

    @pytest.fixture()
    def provider(self) -> CPUFallbackProvider:
        return CPUFallbackProvider()

    @pytest.fixture()
    def onnx_model_path(self, tmp_path: Path) -> str:
        """Create a minimal ONNX model that computes identity (output = input)."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model_path = tmp_path / "identity.onnx"
        onnx.save(model, str(model_path))
        return str(model_path)

    def test_load_model_returns_model_id(
        self, provider: CPUFallbackProvider, onnx_model_path: str,
    ) -> None:
        model_id = provider.load_model(onnx_model_path, {})
        assert model_id == "cpu-0"

    def test_load_nonexistent_file_raises(
        self, provider: CPUFallbackProvider,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            provider.load_model("/nonexistent/model.onnx", {})

    def test_load_and_infer_roundtrip(
        self, provider: CPUFallbackProvider, onnx_model_path: str,
    ) -> None:
        model_id = provider.load_model(onnx_model_path, {})
        test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = provider.infer(model_id, {"input": test_input})
        assert "output" in result
        np.testing.assert_array_almost_equal(result["output"], test_input)

    def test_unload_model_removes_session(
        self, provider: CPUFallbackProvider, onnx_model_path: str,
    ) -> None:
        model_id = provider.load_model(onnx_model_path, {})
        provider.unload_model(model_id)
        with pytest.raises(KeyError):
            provider.infer(model_id, {})

    def test_multiple_models_get_unique_ids(
        self, provider: CPUFallbackProvider, onnx_model_path: str,
    ) -> None:
        id1 = provider.load_model(onnx_model_path, {})
        id2 = provider.load_model(onnx_model_path, {})
        assert id1 != id2
        assert id1 == "cpu-0"
        assert id2 == "cpu-1"
