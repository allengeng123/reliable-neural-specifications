"""Trace pretrained dense ReLU ONNX graphs without rebuilding their layers.

ONNX Runtime evaluates the original operators, weights, shapes, and dtype.
The instrumented copy exposes each live ReLU input and output as graph outputs.
One sample is evaluated at a time, preserving fixed-batch exports and making
the trace independent of the extraction iterator's batching.
"""

import copy
import gzip
import hashlib
from pathlib import Path

import numpy as np


class OnnxFCN:
    def __init__(self, path):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install ONNX extraction support: pip install '.[onnx]'") from exc
        self.path = Path(path)
        raw = self.path.read_bytes()
        if self.path.suffix == ".gz":
            raw = gzip.decompress(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.graph_model = onnx.load_model_from_string(raw)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in self.graph_model.graph.initializer):
            raise ValueError("Export ONNX with embedded weights; external weight files are not supported")
        onnx.checker.check_model(self.graph_model)
        graph = self.graph_model.graph
        initializers = {t.name for t in graph.initializer}
        inputs = [x for x in graph.input if x.name not in initializers]
        if len(inputs) != 1 or len(graph.output) != 1:
            raise ValueError("FCN extraction requires one data input and one score output")
        self.input_name, self.output_name = inputs[0].name, graph.output[0].name
        shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        if len(shape) < 2 or shape[0] not in (0, 1) or any(d <= 0 for d in shape[1:]):
            raise ValueError("Expected input shape (1 or dynamic batch, static feature dimensions...)")
        self.input_shape = tuple(shape[1:])
        elem_type = inputs[0].type.tensor_type.elem_type
        if elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
            raise ValueError("Expected float32 or float64 model inputs")
        self.dtype = np.float32 if elem_type == onnx.TensorProto.FLOAT else np.float64

        # Only consider nodes that contribute to the selected score output.
        live = {self.output_name}
        nodes = []
        for node in reversed(graph.node):
            if live.intersection(node.output):
                nodes.append(node)
                live.update(node.input)
        nodes.reverse()
        allowed = {"Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "BatchNormalization",
                   "Flatten", "Reshape", "Transpose", "Identity", "Constant", "Cast",
                   "Squeeze", "Unsqueeze", "Shape", "Gather", "Concat", "Relu", "Softmax", "LogSoftmax"}
        unsupported = {n.op_type for n in nodes if n.op_type not in allowed or n.domain not in ("", "ai.onnx")}
        if unsupported:
            raise ValueError(f"Unsupported FCN operators: {sorted(unsupported)}; expected a dense ReLU graph")
        self.relu_nodes = [n for n in nodes if n.op_type == "Relu"]
        if not self.relu_nodes:
            raise ValueError("No ReLU activation sites found; this extractor implements ReLU NAPs")

        instrumented = onnx.shape_inference.infer_shapes(copy.deepcopy(self.graph_model))
        known = {v.name: v for v in [*instrumented.graph.input, *instrumented.graph.output,
                                     *instrumented.graph.value_info]}
        outputs = {v.name for v in instrumented.graph.output}
        for node in self.relu_nodes:
            for name in (node.input[0], node.output[0]):
                if name not in known:
                    raise ValueError(f"ONNX shape inference could not identify ReLU tensor {name}")
                if name not in outputs:
                    instrumented.graph.output.append(copy.deepcopy(known[name]))
                    outputs.add(name)
        onnx.checker.check_model(instrumented)
        self.instrumented_model = instrumented
        self.probe_names = [self.output_name]
        self.probe_names += [n.input[0] for n in self.relu_nodes]
        self.probe_names += [n.output[0] for n in self.relu_nodes]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = ort.InferenceSession(instrumented.SerializeToString(), options, providers=["CPUExecutionProvider"])
        self._original = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        probe = self._session.run(self.probe_names, {self.input_name: np.zeros((1, *self.input_shape), self.dtype)})
        self.sizes = (int(np.prod(self.input_shape)), *[int(v.size) for v in probe[1:1 + len(self.relu_nodes)]], int(probe[0].size))
        self.hidden_size = sum(self.sizes[1:-1])
        offset = 0
        self.trace_layout = []
        for node, width in zip(self.relu_nodes, self.sizes[1:-1]):
            self.trace_layout.append({"name": node.output[0], "width": width, "offset": offset})
            offset += width

    def _inputs(self, inputs):
        values = np.asarray(inputs)
        if values.ndim < 2 or np.prod(values.shape[1:]) != self.sizes[0] or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite samples with {self.sizes[0]} features")
        with np.errstate(over="ignore"):
            values = values.astype(self.dtype).reshape(len(values), *self.input_shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("Inputs overflow the model dtype")
        return values

    def forward(self, inputs):
        scores, pre, _ = self.trace(inputs)
        return scores, pre

    def trace(self, inputs):
        values = self._inputs(inputs)
        scores, before, after = [], [], []
        count = len(self.relu_nodes)
        for value in values:
            result = self._session.run(self.probe_names, {self.input_name: value[None]})
            scores.append(result[0].reshape(-1))
            before.append(np.concatenate([v.reshape(-1) for v in result[1:1 + count]]))
            after.append(np.concatenate([v.reshape(-1) for v in result[1 + count:]]))
        if not len(values):
            return (np.empty((0, self.sizes[-1]), self.dtype),
                    np.empty((0, self.hidden_size), self.dtype), np.empty((0, self.hidden_size), self.dtype))
        return np.stack(scores), np.stack(before), np.stack(after)

    def original_scores(self, inputs):
        """Evaluate the untouched ONNX graph for instrumentation parity checks."""
        return np.stack([self._original.run([self.output_name], {self.input_name: x[None]})[0].reshape(-1)
                         for x in self._inputs(inputs)])
