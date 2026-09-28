"""Fetch pinned public assets, then audit FCN extraction on a compute node.

Usage: python scripts/validate_extraction.py fetch
       python scripts/validate_extraction.py run --output results/extraction-audit.json
"""

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from time import perf_counter
import urllib.request

import numpy as np

from neural_specs.cli import environment
from neural_specs.extraction import array_batches, extract
from neural_specs.nnet import NNet
from neural_specs.onnx_fcn import OnnxFCN
from neural_specs.patterns import coverage, write_json


MNIST_COMMIT = "ac970c9360423840f9f83a294cff55672fda454c"
ACAS_COMMIT = "d9e7b2feeaf534dcc557729ee7c920bd5b97099c"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def sources():
    files = []
    for depth in (2, 4, 6):
        name = f"mnist-net_256x{depth}.onnx.gz"
        files.append((name, f"https://raw.githubusercontent.com/VNN-COMP/vnncomp2022_benchmarks/{MNIST_COMMIT}/benchmarks/mnist_fc/onnx/{name}"))
    for i in range(1, 6):
        for j in range(1, 10):
            name = f"ACASXU_run2a_{i}_{j}_batch_2000.onnx.gz"
            files.append((name, f"https://raw.githubusercontent.com/VNN-COMP/vnncomp2023_benchmarks/{ACAS_COMMIT}/benchmarks/acasxu/onnx/{name}"))
    files.extend((name, f"https://ossci-datasets.s3.amazonaws.com/mnist/{name}") for name in MNIST_FILES)
    return files


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root / "sources.json").read_text()) if (root / "sources.json").exists() else {}
    manifest = {}
    for name, url in sources():
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if name in previous and previous[name]["sha256"] != digest:
            raise ValueError(f"Cached asset checksum mismatch: {name}")
        if name in MNIST_FILES and hashlib.md5(content).hexdigest() != MNIST_FILES[name]:
            raise ValueError(f"MNIST standard checksum mismatch: {name}")
        manifest[name] = {"url": url, "sha256": digest}
        print(f"Fetched {name}", flush=True)
    write_json(root / "sources.json", manifest)
    for split, limit in (("train", 1024), ("test", 256)):
        images, labels = mnist(root, split, limit)
        np.savez_compressed(root / f"mnist-{split}-{limit}.npz", inputs=images, labels=labels)


def mnist(root, split, limit):
    prefix = "train" if split == "train" else "t10k"
    image_bytes = gzip.decompress((root / f"{prefix}-images-idx3-ubyte.gz").read_bytes())
    label_bytes = gzip.decompress((root / f"{prefix}-labels-idx1-ubyte.gz").read_bytes())
    magic, count, rows, columns = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if (magic, label_magic, rows, columns) != (2051, 2049, 28, 28) or count != label_count:
        raise ValueError("Malformed MNIST IDX data")
    if len(image_bytes) != 16 + count * 784 or len(label_bytes) != 8 + count:
        raise ValueError("Truncated MNIST IDX data")
    # Matches the benchmark's torchvision ToTensor(), without extra normalization.
    images = np.frombuffer(image_bytes, np.uint8, offset=16).reshape(count, 784)[:limit].astype(np.float32) / np.float32(255)
    labels = np.frombuffer(label_bytes, np.uint8, offset=8)[:limit].astype(np.int64)
    return images, labels


def reference_patterns(postactivations, labels, classes, delta):
    """Independent scalar Algorithm 1, from post-ReLU outputs and exact ratios."""
    threshold = Fraction(str(delta))
    result = []
    for label in range(classes):
        rows = postactivations[labels == label]
        if len(rows) == 0:
            continue
        indices, states = [], []
        for neuron in range(rows.shape[1]):
            count = sum(int(value > 0) for value in rows[:, neuron])
            ratio = Fraction(count, len(rows))
            if ratio >= threshold or ratio <= 1 - threshold:
                indices.append(neuron)
                states.append(int(ratio >= threshold))
        result.append((label, tuple(indices), tuple(states), len(rows)))
    return result


def audit_onnx(path, inputs, supplied_labels, decision, holdout=None):
    from onnx.reference import ReferenceEvaluator
    model = OnnxFCN(path)
    scores, pre, post = model.trace(inputs)
    np.testing.assert_array_equal(pre > 0, post > 0)
    original = model.original_scores(inputs[:16])
    np.testing.assert_array_equal(scores[:16], original)
    # A separate interpreter independently checks every ReLU on 16 real inputs.
    reference = ReferenceEvaluator(model.instrumented_model)
    reference_max_error = 0.0
    for i, sample in enumerate(model._inputs(inputs[:16])):
        values = reference.run(model.probe_names, {model.input_name: sample[None]})
        ref_pre = np.concatenate([v.reshape(-1) for v in values[1:1 + len(model.relu_nodes)]])
        np.testing.assert_array_equal(pre[i] > 0, ref_pre > 0)
        np.testing.assert_allclose(scores[i], values[0].reshape(-1), rtol=1e-4, atol=2e-5)
        reference_max_error = max(reference_max_error, float(np.max(np.abs(pre[i] - ref_pre))))
    reports = {}
    for source in (["predicted", "provided"] if supplied_labels is not None else ["predicted"]):
        labels = supplied_labels if source == "provided" else getattr(scores, decision)(axis=1)
        first = None
        for batch_size in (1, 31, 256):
            result = extract(model, array_batches(inputs, supplied_labels, batch_size),
                             label_source=source, decision=decision, delta="0.95")
            if first is None:
                first = result
            else:
                np.testing.assert_array_equal(result.counts.active, first.counts.active)
                np.testing.assert_array_equal(result.counts.support, first.counts.support)
        for delta in ("0.8", "0.9", "0.95", "0.99", "1.0"):
            patterns = first.counts.patterns(delta)
            actual = [(p.label, p.indices, p.states, p.support) for p in patterns]
            assert actual == reference_patterns(post, labels, model.sizes[-1], delta)
        reports[source] = {**first.metadata, "oracle_patterns_equal": True,
                           "batch_sizes_equal": [1, 31, 256],
                           "neurons_per_class": {str(p.label): len(p.indices) for p in first.patterns}}
        if holdout is not None:
            hx, hy = holdout
            hs, hp = model.forward(hx)
            held_labels = hy if source == "provided" else getattr(hs, decision)(axis=1)
            reports[source]["holdout"] = coverage(first.patterns, hp, held_labels)
    return {"model": path.name, "sha256": model.sha256, "layer_sizes": list(model.sizes),
            "trace_layout": model.trace_layout, "decision": decision, "samples": len(inputs),
            "pre_post_activation_signs_equal": True, "unmodified_onnx_scores_equal": True,
            "independent_interpreter_samples": min(len(inputs), 16),
            "independent_interpreter_signs_equal": True, "reference_max_preactivation_error": reference_max_error,
            "extraction": reports}


def run(root, output):
    started = perf_counter()
    manifest = json.loads((root / "sources.json").read_text())
    for name, info in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Asset changed since fetch: {name}")
    x, y = mnist(root, "train", 1024)
    holdout = mnist(root, "test", 256)
    reports = []
    for depth in (2, 4, 6):
        report = audit_onnx(root / f"mnist-net_256x{depth}.onnx.gz", x, y, "argmax", holdout)
        reports.append(report)
        print(f"PASS {report['model']}", flush=True)
    nnet = NNet.load("data/acasxu/ACASXU_run2a_1_1_batch_2000.nnet")
    lo, hi = nnet.normalized_bounds
    ax = np.random.default_rng(2023).uniform(lo, hi, (64, 5)).astype(np.float32)
    for name, _ in sources():
        if not name.startswith("ACASXU"):
            continue
        report = audit_onnx(root / name, ax, None, "argmin")
        reports.append(report)
        print(f"PASS {name}", flush=True)
    # Cross-format consistency for the bundled ACAS Xu network.
    onnx_model = OnnxFCN(root / "ACASXU_run2a_1_1_batch_2000.onnx.gz")
    onnx_scores, onnx_pre = onnx_model.forward(ax)
    nnet_scores, nnet_pre = nnet.forward(ax)
    np.testing.assert_allclose(onnx_scores, nnet_scores, rtol=1e-4, atol=2e-5)
    np.testing.assert_array_equal(onnx_pre > 0, nnet_pre > 0)
    nnet_extract = extract(nnet, array_batches(ax, batch_size=17), label_source="predicted", decision="argmin")
    onnx_extract = extract(onnx_model, array_batches(ax, batch_size=17), label_source="predicted", decision="argmin")
    assert nnet_extract.patterns == onnx_extract.patterns
    report = {"status": "passed", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "environment": environment(), "seconds": perf_counter() - started,
              "mnist_source": {"year": 2022, "commit": MNIST_COMMIT, "training_samples": 1024,
                               "holdout_samples": 256, "selection": "first examples of official train/test splits",
                               "preprocessing": "float32 pixels / 255; no mean/std normalization"},
              "acas_source": {"year": 2023, "commit": ACAS_COMMIT, "models": 45, "samples_per_model": 64, "seed": 2023},
              "cross_format_acas": {"nnet_onnx_signs_equal": True, "patterns_equal": True,
                                    "maximum_score_error": float(np.max(np.abs(onnx_scores - nnet_scores)))},
              "assets": manifest, "models": reports}
    write_json(output, report)
    print(f"PASS all 48 pretrained FCNs; report: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "run"))
    parser.add_argument("--cache", type=Path, default=Path("benchmarks/cache"))
    parser.add_argument("--output", type=Path, default=Path("results/extraction-audit.json"))
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.cache)
    else:
        run(args.cache, args.output)
