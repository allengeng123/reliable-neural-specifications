"""Algorithm 1: mine per-class activation frequencies, then select stable states."""

from dataclasses import asdict, dataclass
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

    def matches(self, preactivations, margin=0.0):
        values = np.asarray(preactivations)[:, self.indices]
        states = np.asarray(self.states, dtype=bool)
        # Zero belongs to the closed active and inactive half-spaces at margin=0.
        return np.all(np.where(states, values >= margin, values <= -margin), axis=1)


def mine(preactivations, labels, output_size, delta=0.95):
    if not np.isfinite(delta) or not 0.5 < delta <= 1:
        raise ValueError("delta must satisfy 0.5 < delta <= 1")
    values, labels = np.asarray(preactivations), np.asarray(labels)
    if values.ndim != 2 or labels.shape != (len(values),) or len(values) == 0:
        raise ValueError("Expected a nonempty activation matrix and one label per row")
    if not np.all(np.isfinite(values)) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Activations must be finite and labels must be integers")
    if np.any(labels < 0) or np.any(labels >= output_size):
        raise ValueError("Labels outside output range")
    result = []
    for label in range(output_size):
        selected = values[labels == label]
        if not len(selected):
            continue  # Missing classes are reported, never assigned an empty universal NAP.
        frequency = (selected > 0).mean(axis=0)
        active, inactive = frequency >= delta, frequency <= 1 - delta
        indices = np.flatnonzero(active | inactive)
        result.append(Pattern(label, tuple(indices.tolist()), tuple(active[indices].astype(int).tolist()), len(selected)))
    return result


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
               "patterns": [asdict(p) for p in patterns]}
    write_json(path, payload)


def load_patterns(path, model):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("model_sha256") != model.sha256:
        raise ValueError("Pattern schema or model SHA-256 mismatch; mine patterns for this model")
    if data.get("layer_sizes") != list(model.sizes):
        raise ValueError("Pattern/model architecture mismatch")
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
