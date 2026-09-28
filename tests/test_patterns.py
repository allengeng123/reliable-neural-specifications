import json

import numpy as np
import pytest

from neural_specs.patterns import Pattern, coverage, load_patterns, mine, save_patterns


def test_frequency_thresholds_and_missing_classes():
    pre = np.array([[1, -1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1]])
    patterns = mine(pre, np.zeros(4, dtype=int), 2, delta=0.75)
    assert patterns == [Pattern(0, (0, 1), (1, 0), 4)]
    stats = coverage(patterns, pre, np.array([0, 0, 1, 0]))[0]
    assert stats["matched"] == 3
    assert stats["precision"] == pytest.approx(2 / 3)
    assert stats["recall"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("delta", [0, 0.5, 1.01, float("nan")])
def test_invalid_delta(delta):
    with pytest.raises(ValueError):
        mine(np.ones((1, 2)), np.array([0]), 2, delta)


def test_one_example_keeps_neuron_axis():
    assert mine(np.array([[1, -1]]), np.array([0]), 2)[0] == Pattern(0, (0, 1), (1, 0), 1)


def test_pattern_roundtrip_and_model_binding(tmp_path, tiny_model):
    path = tmp_path / "patterns.json"
    pattern = Pattern(0, (0, 1), (1, 0), 10)
    save_patterns(path, tiny_model, [pattern], {"delta": 1})
    assert load_patterns(path, tiny_model) == [pattern]
    payload = json.loads(path.read_text())
    payload["model_sha256"] = "different"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        load_patterns(path, tiny_model)


@pytest.mark.parametrize("pattern", [Pattern(0, (0,), (2500,), 1), Pattern(0, (5,), (1,), 1),
                                      Pattern(0, (0, 0), (0, 1), 1), Pattern(0, (), (), 0)])
def test_invalid_pattern_rejected(pattern):
    with pytest.raises(ValueError):
        pattern.validate(5, 2)


def test_closed_boundary_and_margin():
    active, inactive = Pattern(0, (0,), (1,), 1), Pattern(1, (0,), (0,), 1)
    assert active.matches([[0]])[0] and inactive.matches([[0]])[0]
    assert not active.matches([[0]], margin=0.001)[0]
    assert not inactive.matches([[0]], margin=0.001)[0]
