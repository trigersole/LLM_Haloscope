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

