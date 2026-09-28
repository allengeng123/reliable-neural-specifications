"""Model-independent, streaming NAP extraction from supplied data."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .patterns import ActivationCounts, Pattern


class TraceModel(Protocol):
    """Implement forward as (scores[N,C], pre-ReLU activations[N,H])."""
    sizes: tuple[int, ...]
    hidden_size: int

    def forward(self, inputs): ...


@dataclass
class Extraction:
    patterns: list[Pattern]
    counts: ActivationCounts
    metadata: dict


def extract(model: TraceModel, batches, *, label_source, decision, delta=0.95):
    """Consume (inputs, labels-or-None) batches without storing all activations.

    ``provided`` implements S_l grouping by dataset labels. ``predicted``
    groups by model decisions and must not be described as ground truth.
    We never silently discard misclassified examples from provided labels.
    """
    if label_source not in ("provided", "predicted"):
        raise ValueError("label_source must be provided or predicted")
    if decision not in ("argmin", "argmax"):
        raise ValueError("decision must be argmin (ACAS Xu) or argmax (MNIST)")
    counts = ActivationCounts(model.hidden_size, model.sizes[-1])
    for inputs, labels in batches:
        scores, pre = model.forward(inputs)
        scores, pre = np.asarray(scores), np.asarray(pre)
        if scores.shape != (len(inputs), model.sizes[-1]) or not np.all(np.isfinite(scores)):
            raise ValueError("Model returned invalid scores")
        if label_source == "predicted":
            labels = getattr(scores, decision)(axis=1)
        elif labels is None:
            raise ValueError("Provided-label extraction requires dataset labels")
        counts.update(pre, labels)
    patterns = counts.patterns(delta)
    return Extraction(patterns, counts, {
        "algorithm": "Algorithm 1; exact decimal count cutoffs",
        "activation_rule": "preactivation > 0; zero is inactive",
        "delta": str(delta), "label_source": label_source, "decision": decision,
        "samples": int(counts.support.sum()), "class_support": counts.support.tolist(),
        "missing_labels": np.flatnonzero(counts.support == 0).tolist(),
    })


def array_batches(inputs, labels=None, batch_size=256):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if labels is not None and len(inputs) != len(labels):
        raise ValueError("Each sample must have exactly one label")
    for start in range(0, len(inputs), batch_size):
        yield inputs[start:start + batch_size], None if labels is None else labels[start:start + batch_size]


def load_model(path):
    from pathlib import Path
    path = Path(path)
    if path.suffix == ".nnet":
        from .nnet import NNet
        return NNet.load(path)
    if path.name.endswith((".onnx", ".onnx.gz")):
        from .onnx_fcn import OnnxFCN
        return OnnxFCN(path)
    raise ValueError("Use a .nnet, .onnx, or .onnx.gz pretrained ReLU FCN")
