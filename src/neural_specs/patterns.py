"""Algorithm 1: mine per-class activation frequencies, then select stable states."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pattern:
    label: int
    indices: tuple[int, ...]
    states: tuple[int, ...]
    support: int

    def validate(self, hidden_size, output_size):
        if type(self.label) is not int or not 0 <= self.label < output_size:
            raise ValueError("Invalid pattern label")
        if type(self.support) is not int or self.support <= 0:
            raise ValueError("A mined pattern needs positive class support")
        if len(self.indices) != len(self.states) or len(set(self.indices)) != len(self.indices):
            raise ValueError("Pattern indices/states must align and indices must be unique")
        if any(type(i) is not int or not 0 <= i < hidden_size for i in self.indices):
            raise ValueError("Pattern contains an invalid neuron index")
        if any(type(s) is not int or s not in (0, 1) for s in self.states):
            raise ValueError("Pattern states must be binary, not activation counts")

    def matches(self, preactivations, margin=0.0, *, closed=False):
        """Empirical NAP membership uses >0 / <=0, as in Definition 3.1.

        ``closed=True`` is reserved for the verifier's conservative closure.
        A positive margin restricts both states away from zero.
        """
        values = np.asarray(preactivations)[:, self.indices]
        states = np.asarray(self.states, dtype=bool)
        active = values >= margin if closed or margin > 0 else values > 0
        return np.all(np.where(states, active, values <= -margin), axis=1)


class ActivationCounts:
    """Streaming sufficient statistics, independent of model, depth, or widths."""

    def __init__(self, hidden_size, output_size):
        if type(hidden_size) is not int or hidden_size < 0 or type(output_size) is not int or output_size < 1:
            raise ValueError("Invalid hidden/output dimensions")
        self.active = np.zeros((output_size, hidden_size), dtype=np.int64)
        self.support = np.zeros(output_size, dtype=np.int64)

    def update(self, preactivations, labels):
        values, labels = np.asarray(preactivations), np.asarray(labels)
        if values.ndim != 2 or values.shape[1] != self.active.shape[1] or labels.shape != (len(values),):
            raise ValueError("Expected an activation matrix and one label per row")
        if not np.all(np.isfinite(values)) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Activations must be finite and labels must be integers")
        if np.any(labels < 0) or np.any(labels >= len(self.support)):
            raise ValueError("Labels outside output range")
        for label in np.unique(labels):
            selected = values[labels == label]
            self.active[label] += np.count_nonzero(selected > 0, axis=0)
            self.support[label] += len(selected)

    def patterns(self, delta=0.95):
        # Interpret the supplied decimal exactly: 1 - float(0.9) is below 0.1.
        # Compute integer cutoffs with Python integers, avoiding roundoff and
        # overflow from multiplying an int64 count by a large denominator.
        try:
            fraction = Fraction(str(delta))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("delta must satisfy 0.5 < delta <= 1") from exc
        if not Fraction(1, 2) < fraction <= 1:
            raise ValueError("delta must satisfy 0.5 < delta <= 1")
        if not self.support.sum():
            raise ValueError("Cannot mine patterns from an empty dataset")
        numerator, denominator = fraction.numerator, fraction.denominator
        result = []
        for label, support in enumerate(self.support):
            n = int(support)
            if not n:
                continue
            active_cutoff = (numerator * n + denominator - 1) // denominator
            inactive_cutoff = ((denominator - numerator) * n) // denominator
            active = self.active[label] >= active_cutoff
            inactive = self.active[label] <= inactive_cutoff
            indices = np.flatnonzero(active | inactive)
            result.append(Pattern(label, tuple(indices.tolist()), tuple(active[indices].astype(int).tolist()), n))
        return result


def mine(preactivations, labels, output_size, delta=0.95):
    values = np.asarray(preactivations)
    if values.ndim != 2:
        raise ValueError("Expected an activation matrix")
    counts = ActivationCounts(values.shape[1], output_size)
    counts.update(values, labels)
    return counts.patterns(delta)


def coverage(patterns, preactivations, labels, margin=0.0):
    reports = []
    for pattern in patterns:
        matches = pattern.matches(preactivations, margin)
        correct = np.asarray(labels) == pattern.label
        matched, true_positive, support = int(matches.sum()), int((matches & correct).sum()), int(correct.sum())
        reports.append({"label": pattern.label, "neurons": len(pattern.indices), "class_support": support,
                        "matched": matched, "correct_matches": true_positive,
                        "precision": true_positive / matched if matched else None,
                        "recall": true_positive / support if support else None})
    return reports


def save_patterns(path, model, patterns, metadata):
    payload = {"schema_version": 1, "model_sha256": model.sha256,
               "layer_sizes": list(model.sizes), "metadata": metadata,
               "trace_layout": getattr(model, "trace_layout", None),
               "patterns": [asdict(p) for p in patterns]}
    write_json(path, payload)


def load_patterns(path, model):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("model_sha256") != model.sha256:
        raise ValueError("Pattern schema or model SHA-256 mismatch; mine patterns for this model")
    if data.get("layer_sizes") != list(model.sizes):
        raise ValueError("Pattern/model architecture mismatch")
    if data.get("trace_layout") is not None and data["trace_layout"] != model.trace_layout:
        raise ValueError("Pattern/model neuron layout mismatch")
    patterns = [Pattern(p["label"], tuple(p["indices"]), tuple(p["states"]), p["support"])
                for p in data["patterns"]]
    for pattern in patterns:
        pattern.validate(model.hidden_size, model.sizes[-1])
    if len({p.label for p in patterns}) != len(patterns):
        raise ValueError("Duplicate pattern label")
    return patterns


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
