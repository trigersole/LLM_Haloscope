import numpy as np

from haloscope.splitting import make_split


def test_split_is_deterministic_partition():
    first = make_split(817, seed=41)
    second = make_split(817, seed=41)
    np.testing.assert_array_equal(first.wild, second.wild)
    assert len(first.validation) == 100
    assert len(first.test) == 205
    assert len(set(np.concatenate([first.wild, first.validation, first.test]))) == 817

