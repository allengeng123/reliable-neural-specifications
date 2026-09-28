"""Reproducible, bounded ACAS Xu demonstrations and standalone pipeline commands."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np

from .nnet import NNet
from .patterns import coverage, load_patterns, mine, save_patterns, write_json


DEFAULT_MODEL = "data/acasxu/ACASXU_run2a_1_1_batch_2000.nnet"


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "maraboupy", "onnx", "onnxruntime", "reliable-neural-specifications"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def mining_run(model, samples, holdout, seed, delta, batch_size=256):
    """Uniform physical-domain sampling is equivalent after affine normalization."""
    rng = np.random.default_rng(seed)
    lo, hi = model.normalized_bounds
    inputs = rng.uniform(lo, hi, size=(samples + holdout, model.sizes[0]))
    scores, pre = [], []
    for start in range(0, len(inputs), batch_size):
        output, trace = model.forward(inputs[start:start + batch_size])
        scores.append(output)
        pre.append(trace)
    scores, pre = np.concatenate(scores), np.concatenate(pre)
    labels = scores.argmin(axis=1)
    patterns = mine(pre[:samples], labels[:samples], model.sizes[-1], delta)
    metadata = {"seed": seed, "delta": delta, "mining_samples": samples, "holdout_samples": holdout,
                "sampling": "uniform over the .nnet normalized input domain",
                "labels": "model argmin predictions, not external ground truth",
                "missing_labels": sorted(set(range(model.sizes[-1])) - {p.label for p in patterns}),
                "training": coverage(patterns, pre[:samples], labels[:samples]),
                "holdout": coverage(patterns, pre[samples:], labels[samples:]),
                "environment": environment()}
    return patterns, metadata, inputs[samples:], scores[samples:], pre[samples:]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACAS Xu NAPs — ICML 2023 Oral: Towards Reliable Neural Specifications")
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract", help="extract NAPs from any supported pretrained ReLU FCN and supplied dataset")
    extraction.add_argument("--model", required=True)
    extraction.add_argument("--data", type=Path, required=True, help="NPZ with inputs and optional labels; pickle is disabled")
    extraction.add_argument("--label-source", choices=("provided", "predicted"), required=True)
    extraction.add_argument("--decision", choices=("argmin", "argmax"), required=True)
    extraction.add_argument("--input-space", choices=("model", "physical"), default="model")
    extraction.add_argument("--delta", default="0.95", help="decimal threshold, 0.5 < delta <= 1")
    extraction.add_argument("--batch-size", type=_positive, default=256)
    extraction.add_argument("--output", type=Path, default=Path("results/extract"))
    for command in ("mine", "demo", "verify"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--model", default=DEFAULT_MODEL, help="path to a .nnet model (default relative to repository root)")
        cmd.add_argument("--output", type=Path, default=Path(f"results/{command}"))
        if command in ("mine", "demo"):
            cmd.add_argument("--samples", type=_positive, default=2048)
            cmd.add_argument("--holdout", type=_positive, default=512)
            cmd.add_argument("--seed", type=int, default=2023)
            cmd.add_argument("--delta", type=float, default=0.95)
        if command in ("demo", "verify"):
            cmd.add_argument("--timeout", type=_positive, default=30, help="seconds per solver query")
            cmd.add_argument("--activation-margin", type=float, default=0.0)
            cmd.add_argument("--output-margin", type=float, default=0.0)
        if command == "demo":
            cmd.add_argument("--radius", type=float, default=0.001, help="normalized L-infinity radius for the local NAP demonstration")
        if command == "verify":
            cmd.add_argument("--patterns", type=Path, required=True)
            cmd.add_argument("--label", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            from .extraction import array_batches, extract, load_model
            import hashlib
            model = load_model(args.model)
            with np.load(args.data, allow_pickle=False) as dataset:
                if "inputs" not in dataset:
                    raise ValueError("NPZ must contain an inputs array")
                inputs = dataset["inputs"]
                labels = dataset["labels"] if "labels" in dataset else None
            if args.input_space == "physical":
                if not isinstance(model, NNet):
                    raise ValueError("Physical input normalization is only defined by .nnet headers")
                inputs = model.normalize(inputs)
            result = extract(model, array_batches(inputs, labels, args.batch_size),
                             label_source=args.label_source, decision=args.decision, delta=args.delta)
            result.metadata.update({"environment": environment(), "input_space": args.input_space,
                                    "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                                    "preprocessing": "caller supplies model-ready inputs; physical .nnet inputs are normalized only when requested"})
            save_patterns(args.output / "patterns.json", model, result.patterns, result.metadata)
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output / "counts.npz", active=result.counts.active, support=result.counts.support)
            print(json.dumps(result.metadata, indent=2))
            return 0
        model = NNet.load(args.model)
        if model.sizes[0] != 5 or model.sizes[-1] != 5:
            raise ValueError("This CLI is for ACAS Xu models with five inputs and five outputs")
        if args.command in ("mine", "demo"):
            patterns, metadata, inputs, scores, pre = mining_run(model, args.samples, args.holdout, args.seed, args.delta)
            save_patterns(args.output / "patterns.json", model, patterns, metadata)
            if args.command == "mine":
                print(json.dumps(metadata, indent=2))
                return 0
        from .verification import check_inference, verify
        bounds = None
        report = {"model_sha256": model.sha256, "environment": environment()}
        if args.command == "demo":
            if not np.isfinite(args.radius) or args.radius <= 0:
                raise ValueError("radius must be positive and finite")
            report["mining"] = metadata
            report["inference_check"] = check_inference(model, inputs[:8])
            # Deterministic first held-out input that matches its predicted-class NAP.
            candidates = {p.label: p for p in patterns}
            for i, label in enumerate(scores.argmin(axis=1)):
                pattern = candidates.get(int(label))
                if pattern is not None and pattern.matches(pre[i:i + 1], args.activation_margin)[0]:
                    break
            else:
                raise ValueError("No held-out input matches a mined NAP; adjust delta or increase samples")
            lo, hi = model.normalized_bounds
            bounds = np.maximum(lo, inputs[i] - args.radius), np.minimum(hi, inputs[i] + args.radius)
            report.update({"scope": "local box intersected with a statistically mined NAP; not the full input domain",
                           "anchor_selection": "first matching held-out input in generated order",
                           "holdout_index": i, "anchor": inputs[i].tolist(), "radius": args.radius})
        else:
            patterns = load_patterns(args.patterns, model)
            pattern = next((p for p in patterns if p.label == args.label), None)
            if pattern is None:
                raise ValueError(f"No mined pattern for label {args.label}")
            report["scope"] = "full .nnet input domain intersected with a mined NAP"
        report["verification"] = verify(model, pattern, bounds, args.timeout, args.activation_margin, args.output_margin)
        write_json(args.output / "report.json", report)
        print(json.dumps({"status": report["verification"]["status"], "report": str(args.output / "report.json")}, indent=2))
        return 0 if report["verification"]["status"] == "verified" else 2
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
