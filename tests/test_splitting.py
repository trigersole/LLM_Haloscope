import numpy as np

from haloscope.splitting import make_split


def test_split_is_deterministic_partition():
    first = make_split(817, seed=41)
    second = make_split(817, seed=41)
    np.testing.assert_array_equal(first.wild, second.wild)
    assert len(first.validation) == 100
    assert len(first.test) == 205
    assert len(set(np.concatenate([first.wild, first.validation, first.test]))) == 817
    np.testing.assert_array_equal(first.wild[:10], [0, 1, 3, 9, 11, 13, 14, 15, 16, 17])
    np.testing.assert_array_equal(first.validation[:10], [7, 8, 28, 30, 33, 34, 38, 46, 53, 55])
    np.testing.assert_array_equal(first.test[:10], [2, 4, 5, 6, 10, 12, 21, 23, 24, 32])
