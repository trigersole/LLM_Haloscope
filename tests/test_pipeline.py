import numpy as np

from haloscope.pipeline import HaloScope, SearchConfig
from haloscope.probe import ProbeConfig
from haloscope.splitting import make_split


def synthetic_data(seed=41):
    rng = np.random.default_rng(seed)
    n, layers, hidden = 220, 5, 24
    truth = (rng.random(n) > 0.3).astype(np.int64)
    embeddings = rng.normal(0, 0.5, size=(n, layers, hidden))
    embeddings[truth == 0, 2, :2] += 6.0
    return embeddings.astype(np.float32), truth


def test_complete_pipeline_and_artifact(tmp_path):
    embeddings, truth = synthetic_data()
    split = make_split(len(truth), validation_size=35)
    detector = HaloScope(
        SearchConfig(
            k_values=(1, 2, 3),
            threshold_quantiles=(0.2, 0.3, 0.4, 0.5),
            orientation="paper",
        ),
        ProbeConfig(backend="logistic", epochs=10),
    ).fit(
        embeddings[split.wild],
        embeddings[split.validation],
        truth[split.validation],
    )
    metrics = detector.evaluate(embeddings[split.test], truth[split.test])
    assert metrics["auroc"] > 0.9
    detector.save(tmp_path / "detector")
    restored = HaloScope.load(tmp_path / "detector")
    np.testing.assert_allclose(
        restored.predict_truthfulness(embeddings[split.test]),
        detector.predict_truthfulness(embeddings[split.test]),
    )


def test_confidence_tails_abstain_and_weight_examples():
    detector = HaloScope(
        SearchConfig(
            pseudo_label_mode="confidence_tails",
            confidence_weighted=True,
        ),
        ProbeConfig(backend="logistic"),
    )
    selected, labels, weights, lower, upper = detector._pseudo_labels(
        np.arange(10, dtype=np.float64), 0.2
    )
    assert selected.sum() == 4
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    assert weights is not None and np.all(weights > 0)
    assert lower < upper


def test_confidence_tail_ensemble_round_trip(tmp_path):
    embeddings, truth = synthetic_data()
    split = make_split(len(truth), validation_size=35)
    detector = HaloScope(
        SearchConfig(
            k_values=(1, 2),
            layers=(2,),
            probe_layers=(2,),
            pseudo_label_mode="confidence_tails",
            tail_fractions=(0.2, 0.3),
            confidence_weighted=True,
            probe_repeats=3,
            validation_folds=3,
            stability_penalty=0.25,
            orientation="paper",
        ),
        ProbeConfig(
            backend="logistic",
            epochs=10,
            balanced_loss=True,
            penalty="l2",
        ),
    ).fit(
        embeddings[split.wild],
        embeddings[split.validation],
        truth[split.validation],
    )
    assert len(detector.probes) == 3
    assert detector.summary.pseudo_ignored > 0
    detector.save(tmp_path / "plus_detector")
    restored = HaloScope.load(tmp_path / "plus_detector")
    assert len(restored.probes) == 3
    np.testing.assert_allclose(
        restored.predict_truthfulness(embeddings[split.test]),
        detector.predict_truthfulness(embeddings[split.test]),
    )

