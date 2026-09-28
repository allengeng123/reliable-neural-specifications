import numpy as np
import pytest

from neural_specs.patterns import Pattern
from neural_specs.verification import build_query, check_inference, verify

pytestmark = pytest.mark.solver
pytest.importorskip("maraboupy")


def test_independent_inference_and_variable_mapping(tiny_model):
    report = check_inference(tiny_model, np.array([[0.2, -0.3], [-0.2, 0.1]]))
    assert report["maximum_absolute_error"] < 1e-8
    # Index 3 is the first neuron in layer 2, not a width-50 ACAS offset.
    network = build_query(tiny_model, Pattern(0, (3,), (1,), 1))
    assert network.lowerBounds[network.nodeTo_b(2, 0)] == 0


def test_verified_nonempty_region(tiny_model):
    result = verify(tiny_model, Pattern(0, (0,), (1,), 1), ([0.1, -1], [0.2, 1]), timeout=5)
    assert result["status"] == "verified"
    assert result["feasibility"]["witness"]["validated"]
    assert result["comparisons"][0]["solver_status"] == "unsat"


def test_sat_counterexample(tiny_model):
    result = verify(tiny_model, Pattern(0, (0,), (1,), 1), timeout=5)
    assert result["status"] == "counterexample"
    witness = result["comparisons"][0]["witness"]
    assert witness["validated"]
    assert witness["normalized_scores"][0] >= 0.5 - 1e-6


def test_empty_pattern_region_is_not_verified(tiny_model):
    result = verify(tiny_model, Pattern(0, (0,), (1,), 1), ([-1, -1], [-0.1, 1]), timeout=5)
    assert result["status"] == "empty_region"
    assert result["comparisons"] == []


def test_tie_is_a_strict_dominance_violation(tiny_model):
    result = verify(tiny_model, Pattern(0, (0,), (1,), 1), ([0.5, 0], [0.5, 0]), timeout=5)
    assert result["status"] == "counterexample"


def test_timeout_cannot_become_verified(tiny_model, monkeypatch):
    monkeypatch.setattr("neural_specs.verification._solve", lambda *args: ("TIMEOUT".lower(), {}, 0.01))
    result = verify(tiny_model, Pattern(0, (), (), 1), timeout=1)
    assert result["status"] == "inconclusive"


def test_out_of_domain_rejected(tiny_model):
    with pytest.raises(ValueError, match="inside"):
        verify(tiny_model, Pattern(0, (), (), 1), ([-2, -1], [1, 1]))
