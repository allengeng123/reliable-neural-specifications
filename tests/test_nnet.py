import numpy as np
import pytest

from neural_specs.nnet import NNet


def test_scores_traces_and_argmin(tiny_model):
    inputs = np.array([[0.2, -0.3], [-0.4, 0.5], [0.8, 0.1]])
    scores, trace = tiny_model.forward(inputs)
    np.testing.assert_allclose(scores, [[0.2, 0.5], [0, 0.5], [0.8, 0.5]])
    np.testing.assert_allclose(trace[0], [0.2, -0.2, -0.3, 0.2, 0])
    assert tiny_model.predict(inputs).tolist() == [0, 0, 1]
    assert tiny_model.hidden_size == 5


def test_normalization_does_not_mutate(tiny_model):
    tiny_model.means[:] = [2, 4, 10]
    tiny_model.ranges[:] = [2, 4, 2]
    original = np.array([[8.0, -8.0]])
    np.testing.assert_allclose(tiny_model.normalize(original), [[-0.5, -1.25]])
    np.testing.assert_equal(original, [[8, -8]])
    scores = np.array([[0, 1]])
    np.testing.assert_equal(tiny_model.denormalize_outputs(scores), [[10, 12]])
    np.testing.assert_equal(scores, [[0, 1]])


def test_bad_inputs(tiny_model):
    for inputs in ([1, 2], [[1, 2, 3]], [[float("nan"), 0]]):
        with pytest.raises(ValueError):
            tiny_model.forward(inputs)


def test_truncated_model(tiny_model):
    tiny_model.path.write_text("2,2,2,2,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Truncated"):
        NNet.load(tiny_model.path)
