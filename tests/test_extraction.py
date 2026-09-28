from fractions import Fraction

import numpy as np
import pytest

from neural_specs.extraction import array_batches, extract
from neural_specs.patterns import ActivationCounts, Pattern, mine


def reference_algorithm(values, labels, classes, delta):
    """Literal independent Algorithm 1 oracle, using exact rational frequencies."""
    threshold = Fraction(str(delta))
    patterns = []
    for label in range(classes):
        selected = [row for row, y in zip(values, labels) if y == label]
        if not selected:
            continue
        indices, states = [], []
        for neuron in range(values.shape[1]):
            count = sum(1 for row in selected if row[neuron] > 0)
            frequency = Fraction(count, len(selected))
            if frequency >= threshold:
                indices.append(neuron)
                states.append(1)
            elif frequency <= 1 - threshold:
                indices.append(neuron)
                states.append(0)
        patterns.append(Pattern(label, tuple(indices), tuple(states), len(selected)))
    return patterns


@pytest.mark.parametrize("delta", ["0.9", "0.8", "0.95", "0.99", "1.0", "0.6666666666666667"])
def test_decimal_boundaries_against_independent_algorithm(delta):
    # Every possible count 0..100 appears, including exact inactive cutoffs.
    values = np.where(np.arange(100)[:, None] < np.arange(101)[None, :], 1.0, 0.0)
    labels = np.zeros(100, dtype=int)
    assert mine(values, labels, 2, delta) == reference_algorithm(values, labels, 2, delta)


def test_point_one_is_inactive_at_delta_point_nine():
    values = np.array([[1.0]] + [[0.0]] * 9)
    assert mine(values, np.zeros(10, dtype=int), 1, 0.9) == [Pattern(0, (0,), (0,), 10)]


@pytest.mark.parametrize("batch_size", [1, 7, 32, 101])
def test_streaming_counts_match_oracle_and_batch_size(batch_size):
    rng = np.random.default_rng(98)
    values = rng.integers(-1, 2, size=(101, 13))
    labels = rng.integers(0, 3, size=101)
    counts = ActivationCounts(13, 4)
    for x, y in array_batches(values, labels, batch_size):
        counts.update(x, y)
    assert counts.patterns("0.8") == reference_algorithm(values, labels, 4, "0.8")
    assert counts.support.sum() == 101
    assert counts.support[3] == 0


def test_provided_labels_and_prediction_conventions(tiny_model):
    x = np.array([[-0.1, 0], [0.9, 0]])
    supplied = np.array([1, 1])
    batches = lambda: array_batches(x, supplied, 1)
    provided = extract(tiny_model, batches(), label_source="provided", decision="argmax")
    minimum = extract(tiny_model, batches(), label_source="predicted", decision="argmin")
    maximum = extract(tiny_model, batches(), label_source="predicted", decision="argmax")
    assert provided.counts.support.tolist() == [0, 2]
    assert minimum.counts.support.tolist() == maximum.counts.support.tolist() == [1, 1]
    # Different signs in the two rows make an accidental argmin/argmax swap visible.
    pre = tiny_model.forward(x)[1]
    assert minimum.patterns == reference_algorithm(pre, [0, 1], 2, 0.95)
    assert maximum.patterns == reference_algorithm(pre, [1, 0], 2, 0.95)


def test_bad_or_missing_labels_and_empty_dataset(tiny_model):
    x = np.zeros((1, 2))
    with pytest.raises(ValueError, match="requires dataset labels"):
        extract(tiny_model, [(x, None)], label_source="provided", decision="argmax")
    with pytest.raises(ValueError, match="empty"):
        extract(tiny_model, [], label_source="predicted", decision="argmax")
    with pytest.raises(ValueError, match="integers"):
        extract(tiny_model, [(x, [0.5])], label_source="provided", decision="argmax")


def test_zero_is_never_empirically_active():
    pattern = Pattern(0, (0,), (1,), 1)
    assert pattern.matches([[0], [-0.0], [1e-30], [-1e-30]]).tolist() == [False, False, True, False]
