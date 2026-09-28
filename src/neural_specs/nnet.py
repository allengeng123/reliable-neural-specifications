"""Double-precision .nnet inference with explicit normalization and ReLU traces."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass
class NNet:
    path: Path
    sha256: str
    sizes: tuple[int, ...]
    minimums: np.ndarray
    maximums: np.ndarray
    means: np.ndarray
    ranges: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    @classmethod
    def load(cls, path: str | Path) -> "NNet":
        path = Path(path)
        raw = path.read_bytes()
        rows = iter(line.strip() for line in raw.decode("utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("//"))

        def row(count, integer=False):
            try:
                values = [float(x) for x in next(rows).split(",") if x.strip()]
            except (StopIteration, ValueError) as exc:
                raise ValueError("Truncated or malformed .nnet file") from exc
            if len(values) != count or not np.all(np.isfinite(values)):
                raise ValueError(f"Expected {count} finite values in .nnet row")
            if integer and any(v != int(v) for v in values):
                raise ValueError("Network dimensions must be integers")
            return np.asarray(values, dtype=int if integer else np.float64)

        layers, inputs, outputs, max_size = row(4, True)
        if min(layers, inputs, outputs, max_size) < 1:
            raise ValueError("Network dimensions must be positive")
        sizes = tuple(int(v) for v in row(layers + 1, True))
        if sizes[0] != inputs or sizes[-1] != outputs or min(sizes) < 1 or max(sizes) > max_size:
            raise ValueError("Inconsistent .nnet dimensions")
        if row(1, True)[0] != 0:
            raise ValueError("Symmetric .nnet models are unsupported")
        minimums, maximums = row(inputs), row(inputs)
        means, ranges = row(inputs + 1), row(inputs + 1)
        if np.any(minimums > maximums) or np.any(ranges <= 0):
            raise ValueError("Invalid input bounds or normalization ranges")
        weights, biases = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            weights.append(np.stack([row(nin) for _ in range(nout)]))
            biases.append(np.asarray([row(1)[0] for _ in range(nout)]))
        if next(rows, None) is not None:
            raise ValueError("Unexpected trailing .nnet data")
        return cls(path, hashlib.sha256(raw).hexdigest(), sizes, minimums, maximums,
                   means, ranges, tuple(weights), tuple(biases))

    @property
    def normalized_bounds(self):
        return ((self.minimums - self.means[:-1]) / self.ranges[:-1],
                (self.maximums - self.means[:-1]) / self.ranges[:-1])

    @property
    def hidden_size(self):
        return sum(self.sizes[1:-1])

    def normalize(self, physical_inputs):
        """Clip physical inputs to the declared domain, without mutating the caller."""
        x = self._inputs(physical_inputs)
        return (np.clip(x, self.minimums, self.maximums) - self.means[:-1]) / self.ranges[:-1]

    def _inputs(self, inputs):
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.sizes[0] or not np.all(np.isfinite(x)):
            raise ValueError(f"Expected finite inputs with shape (batch, {self.sizes[0]})")
        return x

    def forward(self, normalized_inputs):
        """Return normalized scores and pre-ReLU activations in layer-major order."""
        x = self._inputs(normalized_inputs)
        traces = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if i < len(self.weights) - 1:
                traces.append(x)
                x = np.maximum(x, 0)
        trace = np.concatenate(traces, axis=1) if traces else np.empty((len(x), 0))
        return x, trace

    def predict(self, normalized_inputs):
        """ACAS Xu selects the LOWEST score (unlike image classifiers)."""
        return self.forward(normalized_inputs)[0].argmin(axis=1)

    def denormalize_outputs(self, scores):
        return np.asarray(scores) * self.ranges[-1] + self.means[-1]
