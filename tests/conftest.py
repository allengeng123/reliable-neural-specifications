from pathlib import Path

import pytest

from neural_specs.nnet import NNet


@pytest.fixture
def tiny_model(tmp_path):
    # Two hidden layers of unequal widths. On x>=0 the scores are [x, 0.5].
    content = """// Synthetic test network, authored for this repository.
3,2,2,3,
2,3,2,2,
0,
-1,-1,
1,1,
0,0,0,
1,1,1,
1,0,
-1,0,
0,1,
0,
0,
0,
1,0,0,
0,1,0,
0,
0,
1,0,
0,0,
0,
0.5,
"""
    path = tmp_path / "tiny.nnet"
    path.write_text(content, encoding="utf-8")
    return NNet.load(path)
