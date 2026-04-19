"""
Poema ONNX Backend — Open Neural Network Exchange format exporter.

Transpiles FMA chains to:
  - ONNX computation graph (using the onnx Python package)
  - Supports real (FP32/FP64) and complex (split Re/Im channels)
  - Quantization metadata (INT8 hints for TensorRT/OpenVINO)
  - TensorRT, OpenVINO, CoreML, ONNX Runtime compatible
  - Batch inference over arbitrary input shapes

FMA chain → ONNX graph mapping:
  Each FMA step  y = w*x + b  maps to:
    - MatMul(x, [[w]]) + [[b]]   for scalar/vector inputs
    - Mul(x, w_tensor) + b_tensor for vectorised path (faster in ORT)

  The full chain becomes a linear sequence of Mul+Add nodes,
  which ONNX-aware backends (TensorRT etc.) may fuse automatically.

Architecture:
  ONNXBackend(BackendProtocol)
    .compile(fma_seq, ast, domain, precision) -> BackendResult
    .build_model(fma_seq, input_name, output_name, precision) -> onnx.ModelProto
    .save_model(model, path) -> str
    .verify_model(model) -> bool
    .load_and_run(model_path, x) -> np.ndarray  (via onnxruntime)
    .verify_available() -> bool
    .quantize_model(model, quant_type) -> onnx.ModelProto

"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from .protocol import BackendCapabilities, BackendProtocol, BackendResult


# ─────────────────────────────────────────────────────────────────────────────
# ONNX availability helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False

def _check_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ONNXBuildInfo:
    model_path: str = ""
    opset_version: int = 17
    n_stages: int = 0
    precision: str = "fp64"
    input_shape: List[int] = field(default_factory=lambda: [-1])
    has_ort: bool = False
    node_count: int = 0

    def summary(self) -> str:
        return (
            f"ONNXBuild [{self.precision}, depth={self.n_stages}]\n"
            f"  Model path  : {self.model_path}\n"
            f"  Opset       : {self.opset_version}\n"
            f"  Node count  : {self.node_count}  (2 per FMA stage: Mul+Add)\n"
            f"  ORT available: {self.has_ort}\n"
            f"  Input shape : {self.input_shape}"
        )


class ONNXBackend(BackendProtocol):
    """
    Poema ONNX Backend.

    Converts FMA chains to ONNX computation graphs compatible with:
      - ONNX Runtime (CPU + GPU providers)
      - TensorRT (NVIDIA, via ORT TensorRT provider)
      - OpenVINO (Intel, via ORT OpenVINO provider)
      - CoreML (Apple Silicon, via ORT CoreML provider)

    Each FMA step  y = w*x + b  becomes two ONNX nodes: Mul + Add.
    For N stages, the graph has 2N + 2 nodes (plus initializers).

    Batch inference: input shape is [-1] (dynamic batch) by default.
    The graph handles arbitrary [batch] shaped inputs.

    Usage:
        from poema.backends.onnx_backend import ONNXBackend
        backend = ONNXBackend()
        result = backend.compile(fma_seq, ast, precision="fp64")
        y = result.callable_fn(x_array)   # uses ORT if available

        # Direct model access:
        model = backend.build_model(fma_seq, precision="fp64")
        backend.save_model(model, "fma_model.onnx")
        is_valid = backend.verify_model(model)
        print(backend.inspect_model(model))
    """

    _BUILD_DIR = Path(tempfile.gettempdir()) / "poema_onnx"

    # ONNX data types (TensorProto.FLOAT=1, TensorProto.DOUBLE=11)
    _DTYPE_MAP = {
        "fp64":   (11, np.float64),  # TensorProto.DOUBLE = 11
        "double": (11, np.float64),
        "fp32":   (1,  np.float32),  # TensorProto.FLOAT  = 1
        "float":  (1,  np.float32),
    }

    def __init__(self, build_dir: Optional[str] = None, opset: int = 17):
        self._build_dir = Path(build_dir) if build_dir else self._BUILD_DIR
        self._build_dir.mkdir(parents=True, exist_ok=True)
        self._opset = opset

    @property
    def capabilities(self) -> BackendCapabilities:
        has_onnx = _check_onnx()
        has_ort  = _check_ort()
        return BackendCapabilities(
            name="onnx",
            supports_cpu=True,
            supports_gpu=True,
            supports_batched=True,
            supports_gradient=False,
            supports_cpp_emit=False,
            hardware_vendor="generic",
            max_fma_depth=100_000,
            precision_formats=["fp32", "fp64"],
            notes=(
                f"ONNX backend (opset {self._opset}). "
                f"onnx: {'available' if has_onnx else 'not installed'}. "
                f"onnxruntime: {'available' if has_ort else 'not installed'}. "
                "Compatible with TensorRT, OpenVINO, CoreML via ORT providers."
            ),
        )

    def verify_available(self) -> bool:
        return _check_onnx()

    # ── Model construction ───────────────────────────────────────────────────

    def build_model(
        self,
        fma_sequence: List[Any],
        input_name: str = "x",
        output_name: str = "y",
        precision: str = "fp64",
        batch_size: int = -1,
    ):
        """
        Build an onnx.ModelProto from the FMA chain.
        Returns the model (or raises ImportError if onnx not installed).
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        onnx_type, np_dtype = self._DTYPE_MAP.get(
            precision, (1, np.float64))

        n = len(fma_sequence)
        nodes = []
        initializers = []

        # Input tensor description (dynamic batch)
        input_shape = [batch_size]  # -1 = dynamic
        x_type = helper.make_tensor_type_proto(onnx_type, input_shape)
        input_info  = helper.make_tensor_value_info(input_name,  onnx_type, input_shape)
        output_info = helper.make_tensor_value_info(output_name, onnx_type, input_shape)

        current_name = input_name

        for i, instr in enumerate(fma_sequence):
            w = np_dtype(getattr(instr.weight, "real", float(instr.weight)))
            b = np_dtype(getattr(instr.bias,   "real", float(instr.bias)))

            w_name   = f"w_{i}"
            b_name   = f"b_{i}"
            mul_out  = f"mul_{i}" if i < n - 1 or True else output_name
            add_out  = output_name if i == n - 1 else f"add_{i}"

            # Weight and bias as scalar initializers (broadcast automatically)
            initializers.append(
                numpy_helper.from_array(np.array(w, dtype=np_dtype), name=w_name))
            initializers.append(
                numpy_helper.from_array(np.array(b, dtype=np_dtype), name=b_name))

            # Mul node: mul_out = current * w
            nodes.append(helper.make_node(
                "Mul", inputs=[current_name, w_name], outputs=[mul_out],
                name=f"Mul_{i}"))

            # Add node: add_out = mul_out + b
            nodes.append(helper.make_node(
                "Add", inputs=[mul_out, b_name], outputs=[add_out],
                name=f"Add_{i}"))

            current_name = add_out

        graph = helper.make_graph(
            nodes,
            name="poema_fma_graph",
            inputs=[input_info],
            outputs=[output_info],
            initializer=initializers,
        )

        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", self._opset)],
        )
        model.doc_string = (
            f"Poema FMA chain, depth={n}, precision={precision}, "
            f"opset={self._opset}"
        )
        model.domain = "poema.act"
        model.model_version = 240   # 2.4.0

        # Shape inference
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            return inferred
        except Exception:
            return model

    # ── Complex ONNX model ──────────────────────────────────────────────────

    def build_complex_model(
        self,
        fma_sequence: List[Any],
    ):
        """
        Build ONNX model for complex-valued FMA chain.
        Inputs: xr (real), xi (imag) as separate f64 tensors.
        Outputs: yr (real), yi (imag) as separate f64 tensors.
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        n = len(fma_sequence)
        nodes = []
        initializers = []
        shape = [-1]

        def make_f64_info(name):
            return helper.make_tensor_value_info(name, TensorProto.DOUBLE, shape)

        ar_name, ai_name = "xr", "xi"

        for i, instr in enumerate(fma_sequence):
            wr = np.float64(getattr(instr.weight, "real", instr.weight) if not isinstance(instr.weight, complex) else instr.weight.real)
            wi = np.float64(getattr(instr.weight, "imag", 0.0))
            br = np.float64(getattr(instr.bias, "real", instr.bias) if not isinstance(instr.bias, complex) else instr.bias.real)
            bi = np.float64(getattr(instr.bias, "imag", 0.0))

            for name, val in [(f"wr_{i}", wr), (f"wi_{i}", wi),
                              (f"br_{i}", br), (f"bi_{i}", bi)]:
                initializers.append(
                    numpy_helper.from_array(np.array(val), name=name))

            # tr = wr*ar - wi*ai + br
            nodes += [
                helper.make_node("Mul", [ar_name, f"wr_{i}"], [f"wr_ar_{i}"], name=f"MulWrAr_{i}"),
                helper.make_node("Mul", [ai_name, f"wi_{i}"], [f"wi_ai_{i}"], name=f"MulWiAi_{i}"),
                helper.make_node("Sub", [f"wr_ar_{i}", f"wi_ai_{i}"], [f"re_prod_{i}"], name=f"SubRe_{i}"),
                helper.make_node("Add", [f"re_prod_{i}", f"br_{i}"], [f"tr_{i}"], name=f"AddBr_{i}"),
                # ti = wr*ai + wi*ar + bi
                helper.make_node("Mul", [ar_name, f"wi_{i}"], [f"wi_ar_{i}"], name=f"MulWiAr_{i}"),
                helper.make_node("Mul", [ai_name, f"wr_{i}"], [f"wr_ai_{i}"], name=f"MulWrAi_{i}"),
                helper.make_node("Add", [f"wi_ar_{i}", f"wr_ai_{i}"], [f"im_prod_{i}"], name=f"AddIm_{i}"),
                helper.make_node("Add", [f"im_prod_{i}", f"bi_{i}"], [f"ti_{i}"], name=f"AddBi_{i}"),
            ]
            ar_name = f"tr_{i}"
            ai_name = f"ti_{i}"

        yr_out = "yr"
        yi_out = "yi"
        nodes += [
            helper.make_node("Identity", [ar_name], [yr_out], name="OutYr"),
            helper.make_node("Identity", [ai_name], [yi_out], name="OutYi"),
        ]

        graph = helper.make_graph(
            nodes, "poema_complex_fma",
            inputs=[make_f64_info("xr"), make_f64_info("xi")],
            outputs=[make_f64_info("yr"), make_f64_info("yi")],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", self._opset)])
        model.doc_string = f"Poema complex FMA, depth={n}"
        return model

    # ── Files ────────────────────────────────────────────────────────────────

    def save_model(self, model, path: str) -> str:
        """Serialize and save an ONNX model to disk."""
        import onnx
        onnx.save(model, path)
        return path

    def verify_model(self, model) -> bool:
        """Run onnx.checker.check_model; returns True if valid."""
        try:
            import onnx
            onnx.checker.check_model(model)
            return True
        except Exception:
            return False

    def inspect_model(self, model) -> str:
        """Return a human-readable summary of an ONNX model."""
        try:
            graph = model.graph
            lines = [
                "=" * 64,
                f"  ONNX Model: {graph.name}",
                f"  Opset     : {model.opset_import[0].version}",
                f"  Doc       : {model.doc_string}",
                f"  Inputs    : {[i.name for i in graph.input]}",
                f"  Outputs   : {[o.name for o in graph.output]}",
                f"  Nodes     : {len(graph.node)}",
                f"  Init vars : {len(graph.initializer)}",
                "  " + "-" * 62,
                "  Node breakdown:",
            ]
            from collections import Counter
            op_counts = Counter(n.op_type for n in graph.node)
            for op, cnt in sorted(op_counts.items()):
                lines.append(f"    {op:<16}: {cnt}")
            lines.append("=" * 64)
            return "\n".join(lines)
        except Exception as e:
            return f"[ONNX inspect error: {e}]"

    def load_and_run(self, model_path: str, x: np.ndarray) -> np.ndarray:
        """Load ONNX model and run inference via ONNX Runtime."""
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path)
        inp_name = sess.get_inputs()[0].name
        dtype = x.dtype
        result = sess.run(None, {inp_name: x.astype(dtype)})
        return result[0]

    def quantize_model(self, model, quant_type: str = "int8"):
        """
        Apply post-training quantization to the ONNX model.
        quant_type: 'int8' | 'uint8' | 'float16'
        Requires onnxruntime.quantization.
        """
        import tempfile
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_in:
                self.save_model(model, tmp_in.name)
                tmp_in_path = tmp_in.name
            tmp_out = tmp_in_path.replace(".onnx", "_quant.onnx")
            qt = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}.get(
                quant_type, QuantType.QInt8)
            quantize_dynamic(tmp_in_path, tmp_out, weight_type=qt)
            import onnx
            return onnx.load(tmp_out)
        except Exception as e:
            return model   # return unquantized on failure

    # ── BackendProtocol interface ────────────────────────────────────────────

    def compile(
        self,
        fma_sequence: List[Any],
        source_ast: Any,
        domain: Tuple[float, float] = (-1.0, 1.0),
        precision: str = "fp64",
        module_name: Optional[str] = None,
        **kwargs,
    ) -> BackendResult:
        is_complex = kwargs.get("complex_mode", False)
        n = len(fma_sequence)
        mod_name = module_name or f"poema_fma_{precision}_{n}"
        model_path = str(self._build_dir / f"{mod_name}.onnx")

        model = None
        emitted_code = None
        callable_fn = None
        node_count = 0

        if _check_onnx():
            try:
                if is_complex:
                    model = self.build_complex_model(fma_sequence)
                else:
                    model = self.build_model(
                        fma_sequence, precision=precision)
                self.save_model(model, model_path)
                emitted_code = self.inspect_model(model)
                node_count = len(model.graph.node)

                if _check_ort():
                    # Callable via ONNX Runtime
                    _, np_dtype = self._DTYPE_MAP.get(precision, (11, np.float64))
                    _path = model_path
                    _dtype = np_dtype
                    def _ort_fn(x: np.ndarray) -> np.ndarray:
                        return self.load_and_run(_path, np.asarray(x, dtype=_dtype))
                    callable_fn = _ort_fn
            except Exception as e:
                emitted_code = f"[ONNX build error: {e}]"

        # Fallback callable: pure numpy
        if callable_fn is None:
            ws = [float(getattr(i.weight, "real", i.weight)) for i in fma_sequence]
            bs = [float(getattr(i.bias,   "real", i.bias))   for i in fma_sequence]
            def _np_fn(x: np.ndarray) -> np.ndarray:
                y = np.asarray(x, dtype=np.float64).copy()
                for w, b in zip(ws, bs):
                    y = w * y + b
                return y
            callable_fn = _np_fn

        return BackendResult(
            callable_fn=callable_fn,
            emitted_code=emitted_code,
            emitted_path=model_path if os.path.exists(model_path) else None,
            fma_count=n,
            backend_name="onnx",
            extra={
                "model_path" : model_path if os.path.exists(model_path) else None,
                "model"      : model,
                "node_count" : node_count,
                "has_onnx"   : _check_onnx(),
                "has_ort"    : _check_ort(),
                "opset"      : self._opset,
                "precision"  : precision,
            },
        )

    # ── Export / conversion helpers ──────────────────────────────────────────

    def to_torch_script(self, fma_sequence: List[Any], precision: str = "fp64") -> str:
        """
        Generate TorchScript (Python) code equivalent to the FMA chain.
        Useful for tracing to TorchScript and then converting to ONNX via PyTorch.
        """
        dtype_str = "torch.float64" if precision in ("fp64", "double") else "torch.float32"
        lines = [
            "import torch",
            "",
            "class PoemFMA(torch.nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        # FMA chain: {len(fma_sequence)} stages, {precision}",
        ]
        for i, instr in enumerate(fma_sequence):
            w = float(getattr(instr.weight, "real", instr.weight))
            b = float(getattr(instr.bias, "real", instr.bias))
            lines += [
                f"        self.w{i} = torch.tensor({w:.17g}, dtype={dtype_str})",
                f"        self.b{i} = torch.tensor({b:.17g}, dtype={dtype_str})",
            ]
        lines += [
            "",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
            "        y = x",
        ]
        for i in range(len(fma_sequence)):
            lines.append(f"        y = self.w{i} * y + self.b{i}")
        lines += ["        return y", ""]
        return "\n".join(lines)
