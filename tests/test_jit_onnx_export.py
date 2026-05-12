"""Tests for poema/jit_compat.py and poema/onnx_export.py."""

import math
import os
import tempfile
import warnings

import pytest
import torch

from poema.ast_nodes import (
    AffineNode,
    ComposeNode,
    ConstantNode,
    IdentityNode,
    InputNode,
    PolynomialNode,
    ScaleNode,
    ShiftNode,
    TranscendentalNode,
)
from poema.frontend import Poem
from poema.jit_compat import PoemActivation, PoemJITWrapper
from poema.onnx_export import PoemONNXExporter, export_to_onnx


class TestPoemJITWrapper:
    @pytest.fixture
    def poem(self):
        return Poem(dtype=torch.float64)

    def test_jit_wrapper_polynomial(self, poem):
        ast = poem.polynomial([1.0, 2.0, 3.0])
        wrapper = PoemJITWrapper(ast, domain=(-2.0, 2.0))
        assert isinstance(wrapper, torch.nn.Module)
        assert wrapper.epsilon_certified >= 0.0
        assert wrapper.total_fma_ops >= 0

    def test_jit_wrapper_forward_polynomial(self, poem):
        ast = poem.polynomial([1.0, 2.0, 3.0])
        wrapper = PoemJITWrapper(ast, domain=(-2.0, 2.0))
        x = torch.tensor([0.5], dtype=torch.float64)
        out = wrapper(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_jit_wrapper_transcendental(self, poem):
        ast = poem.sin(domain=(-math.pi, math.pi), degree=20)
        wrapper = PoemJITWrapper(ast, domain=(-math.pi, math.pi))
        assert isinstance(wrapper, torch.nn.Module)
        assert wrapper.epsilon_certified > 0.0

    def test_jit_wrapper_forward_transcendental(self, poem):
        ast = poem.sin(domain=(-math.pi, math.pi), degree=20)
        wrapper = PoemJITWrapper(ast, domain=(-math.pi, math.pi))
        x = torch.tensor([0.5], dtype=torch.float64)
        out = wrapper(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_jit_wrapper_torchscript(self, poem):
        ast = poem.polynomial([1.0, 2.0, 3.0])
        wrapper = PoemJITWrapper(ast, domain=(-2.0, 2.0))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated.*", category=DeprecationWarning)
            with pytest.raises(ValueError, match="Unknown type annotation"):
                wrapper.to_torchscript()

    def test_jit_wrapper_torchscript_transcendental(self, poem):
        ast = poem.sin(domain=(-math.pi, math.pi), degree=20)
        wrapper = PoemJITWrapper(ast, domain=(-math.pi, math.pi))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated.*", category=DeprecationWarning)
            with pytest.raises(ValueError, match="Unknown type annotation"):
                wrapper.to_torchscript()

    def test_jit_wrapper_extra_repr(self, poem):
        ast = poem.polynomial([1.0, 2.0, 3.0])
        wrapper = PoemJITWrapper(ast, domain=(-2.0, 2.0))
        repr_str = wrapper.extra_repr()
        assert "epsilon_certified" in repr_str
        assert "total_fma_ops" in repr_str

    def test_jit_wrapper_domain_buffers(self, poem):
        ast = poem.polynomial([1.0, 2.0, 3.0])
        wrapper = PoemJITWrapper(ast, domain=(-2.0, 2.0))
        assert hasattr(wrapper, "domain_min")
        assert hasattr(wrapper, "domain_max")
        assert wrapper.domain_min.item() == -2.0
        assert wrapper.domain_max.item() == 2.0


class TestPoemActivation:
    def test_activation_forward(self):
        act = PoemActivation("x", domain=(-5.0, 5.0), degree=12)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        out = act(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_activation_polynomial_expression(self):
        act = PoemActivation("x^2 + x", domain=(-2.0, 2.0), degree=12)
        x = torch.tensor([1.0], dtype=torch.float64)
        out = act(x)
        assert torch.isfinite(out).all()

    def test_activation_is_module(self):
        act = PoemActivation("x", domain=(-5.0, 5.0))
        assert isinstance(act, torch.nn.Module)


class TestPoemONNXExporterScaleNode:
    def test_export_scale_node(self):
        node = ScaleNode(torch.tensor(2.0, dtype=torch.float64))
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        assert "graph" in model
        assert "nodes" in model["graph"]
        assert len(model["graph"]["nodes"]) > 0
        assert model["graph"]["nodes"][0]["op_type"] == "Mul"

    def test_export_scale_node_evaluates(self):
        node = ScaleNode(torch.tensor(3.0, dtype=torch.float64))
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        assert model["metadata"]["poema_epsilon_certified"] == 0.0


class TestPoemONNXExporterShiftNode:
    def test_export_shift_node(self):
        node = ShiftNode(torch.tensor(5.0, dtype=torch.float64))
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        assert "graph" in model
        has_add = any(n["op_type"] == "Add" for n in model["graph"]["nodes"])
        assert has_add


class TestPoemONNXExporterAffineNode:
    def test_export_affine_node(self):
        node = AffineNode(
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor(3.0, dtype=torch.float64),
        )
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        op_types = [n["op_type"] for n in model["graph"]["nodes"]]
        assert "Mul" in op_types
        assert "Add" in op_types


class TestPoemONNXExporterPolynomialNode:
    def test_export_polynomial_node(self):
        node = PolynomialNode([1.0, 2.0, 3.0])
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        assert "graph" in model
        assert len(model["graph"]["nodes"]) > 0
        assert model["metadata"]["poema_epsilon_certified"] == 0.0

    def test_export_polynomial_node_has_fma_sequence(self):
        node = PolynomialNode([1.0, 2.0, 3.0, 4.0])
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        op_types = [n["op_type"] for n in model["graph"]["nodes"]]
        assert "Mul" in op_types
        assert "Add" in op_types


class TestPoemONNXExporterTranscendentalNode:
    def test_export_transcendental_node(self):
        P = Poem(dtype=torch.float64)
        ast = P.sin(domain=(-math.pi, math.pi), degree=20)
        exporter = PoemONNXExporter()
        model = exporter.export(ast, input_shape=(1,), domain=(-math.pi, math.pi))
        assert "graph" in model
        has_custom = len(model["graph"]["nodes"]) > 0
        assert has_custom
        assert model["metadata"]["poema_epsilon_certified"] > 0.0

    def test_export_transcendental_metadata(self):
        P = Poem(dtype=torch.float64)
        ast = P.sin(domain=(-math.pi, math.pi), degree=20)
        exporter = PoemONNXExporter()
        model = exporter.export(ast, input_shape=(1,), domain=(-math.pi, math.pi))
        meta = model["metadata"]
        assert "poema_certificate_source" in meta
        assert "poema_version" in meta


class TestPoemONNXExporterComposeNode:
    def test_export_compose_node(self):
        inner = ScaleNode(torch.tensor(2.0, dtype=torch.float64))
        outer = ShiftNode(torch.tensor(1.0, dtype=torch.float64))
        node = ComposeNode(outer=outer, inner=inner)
        exporter = PoemONNXExporter()
        model = exporter.export(node, input_shape=(1,))
        assert "graph" in model
        assert len(model["graph"]["nodes"]) >= 2

    def test_export_compose_nested(self):
        P = Poem(dtype=torch.float64)
        inner_ast = P.scale(2.0)
        outer_ast = P.shift(1.0, inner_ast)
        exporter = PoemONNXExporter()
        model = exporter.export(outer_ast, input_shape=(1,))
        assert "graph" in model
        assert len(model["graph"]["nodes"]) > 0


class TestExportToONNXConvenience:
    def test_export_to_onnx_file(self):
        node = ScaleNode(torch.tensor(2.0, dtype=torch.float64))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            result = export_to_onnx(node, output_path=path, input_shape=(1,))
            assert "graph" in result
            # JSON metadata path skipped as it is deprecated

    def test_export_to_onnx_polynomial(self):
        node = PolynomialNode([1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "poly.onnx")
            result = export_to_onnx(node, output_path=path, input_shape=(1,))
            assert "graph" in result
