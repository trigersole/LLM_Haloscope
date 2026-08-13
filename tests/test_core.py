import numpy as np

from haloscope.core import LatentSubspace, SubspaceConfig


def test_membership_score_matches_equation_7():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    model = LatentSubspace(SubspaceConfig(n_components=2, weighted=True)).fit(x)
    projected = (x - x.mean(axis=0)) @ model.components_.T
    expected = np.mean(projected**2 * model.singular_values_[None, :], axis=1)
    np.testing.assert_allclose(model.score(x), expected)


def test_subspace_save_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 5))
    model = LatentSubspace(SubspaceConfig(3, weighted=False)).fit(x)
    model.save(tmp_path / "state.npz")
    restored = LatentSubspace.load(tmp_path / "state.npz")
    np.testing.assert_allclose(restored.score(x), model.score(x))

