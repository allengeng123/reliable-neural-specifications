"""Marabou checks for nonempty NAP regions and strict ACAS argmin dominance."""

from time import perf_counter

import numpy as np


def _backend():
    try:
        from maraboupy import Marabou, MarabouCore, MarabouUtils
    except ImportError as exc:
        raise RuntimeError("Install the solver extra: pip install '.[verify]' (Python 3.10/3.11 on Linux)") from exc
    return Marabou, MarabouCore, MarabouUtils


def _bounds(model, bounds):
    domain_lo, domain_hi = model.normalized_bounds
    lo, hi = (domain_lo, domain_hi) if bounds is None else map(lambda x: np.asarray(x, dtype=float), bounds)
    if lo.shape != domain_lo.shape or hi.shape != domain_hi.shape:
        raise ValueError("Bounds must have one entry per input")
    if not np.all(np.isfinite([lo, hi])) or np.any(lo > hi):
        raise ValueError("Bounds must be finite and ordered")
    if np.any(lo < domain_lo) or np.any(hi > domain_hi):
        raise ValueError("Verification bounds must be inside the model's normalized domain")
    return lo, hi


def build_query(model, pattern, bounds=None, activation_margin=0.0):
    pattern.validate(model.hidden_size, model.sizes[-1])
    if not np.isfinite(activation_margin) or activation_margin < 0:
        raise ValueError("activation_margin must be finite and nonnegative")
    lo, hi = _bounds(model, bounds)
    Marabou, _, _ = _backend()
    network = Marabou.read_nnet(str(model.path), normalize=False)
    if tuple(network.layerSizes) != model.sizes:
        raise RuntimeError("Marabou and NumPy disagree about the architecture")
    for variable, lower, upper in zip(np.asarray(network.inputVars).reshape(-1), lo, hi):
        network.setLowerBound(int(variable), float(lower))
        network.setUpperBound(int(variable), float(upper))
    # Derive layer-major preactivation variables from Marabou's actual network.
    variables = [network.nodeTo_b(layer, node)
                 for layer in range(1, len(model.sizes) - 1)
                 for node in range(model.sizes[layer])]
    for index, state in zip(pattern.indices, pattern.states):
        if state:
            network.setLowerBound(variables[index], float(activation_margin))
        else:
            network.setUpperBound(variables[index], float(-activation_margin))
    return network


def _solve(network, timeout):
    Marabou, _, _ = _backend()
    started = perf_counter()
    status, values, _ = network.solve(options=Marabou.createOptions(
        verbosity=0, numWorkers=1, timeoutInSeconds=timeout), verbose=False)
    return str(status).lower(), values, perf_counter() - started


def _witness(model, pattern, network, values, bounds, activation_margin, other=None, output_margin=0.0):
    """Re-evaluate SAT witnesses independently; numerical failures stay inconclusive."""
    x = np.asarray([values[int(v)] for v in np.asarray(network.inputVars).reshape(-1)])
    scores, pre = model.forward(x[None, :])
    lo, hi = _bounds(model, bounds)
    tolerance = 1e-6
    valid = bool(np.all(x >= lo - tolerance) and np.all(x <= hi + tolerance))
    valid = valid and bool(pattern.matches(pre, activation_margin - tolerance)[0])
    if other is not None:
        valid = valid and bool(scores[0, pattern.label] - scores[0, other] >= -output_margin - tolerance)
    return {"normalized_input": x.tolist(), "normalized_scores": scores[0].tolist(),
            "predicted_label": int(scores[0].argmin()), "validated": valid,
            "validation_tolerance": tolerance}


def verify(model, pattern, bounds=None, timeout=30, activation_margin=0.0, output_margin=0.0):
    """Prove y[label] + output_margin < y[j] for every competitor j.

    A SAT violation includes ties; it need not change NumPy's tie-broken argmin.
    First solve the region itself, so an empty NAP never becomes a certificate.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if not np.isfinite(output_margin) or output_margin < 0:
        raise ValueError("output_margin must be finite and nonnegative")
    lo, hi = _bounds(model, bounds)
    network = build_query(model, pattern, (lo, hi), activation_margin)
    status, values, seconds = _solve(network, timeout)
    report = {"label": pattern.label, "neurons": len(pattern.indices),
              "normalized_lower": lo.tolist(), "normalized_upper": hi.tolist(),
              "activation_margin": activation_margin, "output_margin": output_margin,
              "feasibility": {"solver_status": status, "seconds": seconds},
              "comparisons": [], "status": "inconclusive"}
    if status == "unsat":
        report["status"] = "empty_region"
        return report
    if status != "sat":
        return report
    witness = _witness(model, pattern, network, values, (lo, hi), activation_margin)
    report["feasibility"]["witness"] = witness
    if not witness["validated"]:
        return report
    _, MarabouCore, MarabouUtils = _backend()
    for other in range(model.sizes[-1]):
        if other == pattern.label:
            continue
        query = build_query(model, pattern, (lo, hi), activation_margin)
        outputs = np.asarray(query.outputVars).reshape(-1)
        violation = MarabouUtils.Equation(MarabouCore.Equation.GE)
        violation.addAddend(1, int(outputs[pattern.label]))
        violation.addAddend(-1, int(outputs[other]))
        violation.setScalar(-float(output_margin))
        query.addEquation(violation)
        status, values, seconds = _solve(query, timeout)
        comparison = {"other_label": other, "solver_status": status, "seconds": seconds}
        if status == "sat":
            comparison["witness"] = _witness(model, pattern, query, values, (lo, hi), activation_margin, other, output_margin)
        report["comparisons"].append(comparison)
    if all(c["solver_status"] == "unsat" for c in report["comparisons"]):
        report["status"] = "verified"
    elif any(c["solver_status"] == "sat" and c["witness"]["validated"] for c in report["comparisons"]):
        report["status"] = "counterexample"
    return report


def check_inference(model, normalized_inputs):
    """Compare every output score with Marabou's independent .nnet evaluator."""
    Marabou, _, _ = _backend()
    network = Marabou.read_nnet(str(model.path), normalize=False)
    expected = model.forward(normalized_inputs)[0]
    actual = np.asarray([np.asarray(network.evaluateNNet(x.tolist(), normalize_inputs=False,
                                                       normalize_outputs=False)).reshape(-1)
                         for x in normalized_inputs])
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    return {"samples": len(expected), "maximum_absolute_error": float(np.max(np.abs(actual - expected)))}
