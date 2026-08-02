"""CPU fallback accelerator using ONNX Runtime.

The lowest-priority accelerator backend — runs on any machine with
onnxruntime installed. Slow (~100ms/frame) but guaranteed to work
without GPU hardware. Satisfies the AcceleratorHAL protocol.

onnxruntime is lazy-imported so the module can be imported on systems
where the [vision] extra is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CPUFallbackProvider:
    """ONNX Runtime CPU inference backend."""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._model_counter: int = 0

    def load_model(
        self, model_path: str, input_shapes: dict[str, object],
    ) -> str:
        try:
            import onnxruntime as ort
        except ImportError:
            msg = (
                "onnxruntime is not installed. "
                "Install with: pip install aerialmind[vision]"
            )
            raise RuntimeError(msg) from None

        path = Path(model_path)
        if not path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"],
        )
        model_id = f"cpu-{self._model_counter}"
        self._model_counter += 1
        self._sessions[model_id] = session
        return model_id

    def infer(
        self,
        model_id: str,
        inputs: dict[str, NDArray[np.float32]],
    ) -> dict[str, NDArray[np.float32]]:
        if model_id not in self._sessions:
            msg = f"Unknown model_id: {model_id}"
            raise KeyError(msg)

        session = self._sessions[model_id]
        input_feed = {k: v for k, v in inputs.items()}
        output_names = [o.name for o in session.get_outputs()]
        results = session.run(output_names, input_feed)
        return dict(zip(output_names, results))

    def get_capabilities(self) -> dict[str, object]:
        return {
            "precision": ["fp32"],
            "max_batch": 1,
            "device_name": "cpu",
        }

    def unload_model(self, model_id: str) -> None:
        if model_id not in self._sessions:
            msg = f"Unknown model_id: {model_id}"
            raise KeyError(msg)
        del self._sessions[model_id]
