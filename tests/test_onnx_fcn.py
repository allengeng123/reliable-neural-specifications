import hashlib

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from neural_specs.extraction import array_batches, extract
from neural_specs.onnx_fcn import OnnxFCN


def make_model(tmp_path, *, matmul=False, dynamic=False, nonlinear="Relu"):
    rng = np.random.default_rng(42)
    nodes, initializers = [], []
    previous = "input"
    for i, (nin, nout) in enumerate(zip((3, 7, 2), (7, 2, 4))):
        weight = rng.normal(size=(nin, nout)).astype(np.float32)
        bias = rng.normal(size=nout).astype(np.float32)
        initializers.extend([numpy_helper.from_array(weight, f"w{i}"), numpy_helper.from_array(bias, f"b{i}")])
        if matmul:
            nodes.extend([helper.make_node("MatMul", [previous, f"w{i}"], [f"mm{i}"]),
                          helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"pre{i}"])])
        else:
            nodes.append(helper.make_node("Gemm", [previous, f"w{i}", f"b{i}"], [f"pre{i}"]))
        previous = f"pre{i}"
        if i < 2:
            nodes.append(helper.make_node(nonlinear, [previous], [f"post{i}"]))
            previous = f"post{i}"
    batch = "batch" if dynamic else 1
    graph = helper.make_graph(nodes, "unequal-width-fcn",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3])],
                              [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [batch, 4])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("matmul,dynamic", [(False, False), (True, False), (True, True)])
def test_trace_all_layers_and_original_graph_parity(tmp_path, matmul, dynamic):
    path = make_model(tmp_path, matmul=matmul, dynamic=dynamic)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    model = OnnxFCN(path)
    x = np.random.default_rng(6).normal(size=(17, 3)).astype(np.float32)
    scores, pre, post = model.trace(x)
    assert model.sizes == (3, 7, 2, 4)
    assert [r["offset"] for r in model.trace_layout] == [0, 7]
    np.testing.assert_array_equal(pre > 0, post > 0)
    np.testing.assert_array_equal(scores, model.original_scores(x))
    reference = ReferenceEvaluator(model.instrumented_model)
    for i in range(len(x)):
        values = reference.run(model.probe_names, {model.input_name: x[i:i + 1]})
        expected_pre = np.concatenate([v.reshape(-1) for v in values[1:3]])
        np.testing.assert_allclose(scores[i], values[0][0], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(pre[i] > 0, expected_pre > 0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    one = extract(model, array_batches(x, batch_size=1), label_source="predicted", decision="argmax")
    many = extract(model, array_batches(x, batch_size=8), label_source="predicted", decision="argmax")
    np.testing.assert_array_equal(one.counts.active, many.counts.active)
    assert one.patterns == many.patterns


def test_non_relu_fcn_rejected_explicitly(tmp_path):
    with pytest.raises(ValueError, match="Sigmoid"):
        OnnxFCN(make_model(tmp_path, nonlinear="Sigmoid"))


def test_npz_extraction_cli(tmp_path):
    from neural_specs.cli import main
    from neural_specs.patterns import load_patterns
    path = make_model(tmp_path)
    dataset = tmp_path / "data.npz"
    np.savez(dataset, inputs=np.zeros((3, 3), dtype=np.float32), labels=np.array([0, 1, 1]))
    output = tmp_path / "out"
    assert main(["extract", "--model", str(path), "--data", str(dataset), "--label-source", "provided",
                 "--decision", "argmax", "--output", str(output)]) == 0
    assert [p.support for p in load_patterns(output / "patterns.json", OnnxFCN(path))] == [1, 2]
    with np.load(output / "counts.npz") as counts:
        assert counts["support"].tolist() == [1, 2, 0, 0]
