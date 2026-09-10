import numpy as np
import pytest

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


def test_truncated_subspace_matches_direct_fit():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(30, 8))
    full = LatentSubspace(SubspaceConfig(5, weighted=True)).fit(x)
    truncated = full.truncated(2)
    direct = LatentSubspace(SubspaceConfig(2, weighted=True)).fit(x)
    np.testing.assert_allclose(truncated.score(x), direct.score(x))


def test_official_score_matches_released_projection_order():
    x = np.array([[2.0, 1.0], [-1.0, 3.0], [0.5, -2.0]])
    model = LatentSubspace(
        SubspaceConfig(
            2,
            weighted=True,
            center=True,
            score_centered=False,
            score_mode="official",
            deterministic_component_sign=True,
        )
    ).fit(x)
    expected = np.abs(
        np.mean((x @ model.components_.T) * model.singular_values_[None, :], axis=1)
    )
    np.testing.assert_allclose(model.score(x), expected)


def test_sklearn_backend_matches_float32_sklearn_pca():
    sklearn = pytest.importorskip("sklearn.decomposition")
    rng = np.random.RandomState(7)
    x = rng.normal(size=(32, 12)).astype(np.float32)

    np.random.seed(19)
    expected = sklearn.PCA(
        n_components=4, whiten=False, svd_solver="auto", random_state=None
    ).fit(x)
    np.random.seed(19)
    actual = LatentSubspace(
        SubspaceConfig(
            n_components=4,
            weighted=True,
            score_centered=False,
            score_mode="official",
            factorization_backend="sklearn_auto",
        )
    ).fit(x)

    np.testing.assert_allclose(actual.mean_, expected.mean_)
    np.testing.assert_allclose(actual.components_, expected.components_)
    np.testing.assert_allclose(actual.singular_values_, expected.singular_values_)
    projected = x @ expected.components_.T
    expected_scores = np.abs(
        np.mean(projected * expected.singular_values_[None, :], axis=1)
    )
    np.testing.assert_allclose(actual.score(x), expected_scores)
