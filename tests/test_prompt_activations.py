import numpy as np

from haloscope.modeling import HFActivationModel, ModelConfig, render_activation_prompt


def _record(identifier="x"):
    return {
        "id": identifier,
        "question": "Who wrote Hamlet?",
        "context": "Hamlet is a tragedy by William Shakespeare.",
        "prompt": "Q: Who wrote Hamlet? A:",
        "answer": "William Shakespeare.",
    }


def test_render_activation_prompt_supports_context_and_answer():
    rendered = render_activation_prompt(
        "{context_block}Q: {question}\nA: {answer}\nAssessment:", _record()
    )
    assert "Context: Hamlet is a tragedy" in rendered
    assert "Q: Who wrote Hamlet?" in rendered
    assert "A: William Shakespeare." in rendered
    assert rendered.endswith("Assessment:")


def test_contrastive_extraction_is_positive_minus_negative():
    model = HFActivationModel.__new__(HFActivationModel)
    model.config = ModelConfig(model_name="unused", activation_mode="contrastive")

    def fake_extract(texts):
        sign = 2.0 if "factually correct" in texts[0] else 0.5
        return np.full((len(texts), 2, 3), sign, dtype=np.float32)

    model.extract = fake_extract
    result = model.extract_records([_record("a"), _record("b")])
    np.testing.assert_allclose(result, 1.5)
    assert result.shape == (2, 2, 3)


def test_response_extraction_preserves_existing_behavior():
    model = HFActivationModel.__new__(HFActivationModel)
    model.config = ModelConfig(model_name="unused", activation_mode="response")
    captured = []

    def fake_extract(texts):
        captured.extend(texts)
        return np.zeros((len(texts), 1, 2), dtype=np.float32)

    model.extract = fake_extract
    model.extract_records([_record()])
    assert captured == ["Q: Who wrote Hamlet? A:William Shakespeare."]


def test_prompt_extraction_uses_only_original_prompt():
    model = HFActivationModel.__new__(HFActivationModel)
    model.config = ModelConfig(model_name="unused", activation_mode="prompt")
    captured = []

    def fake_extract(texts):
        captured.extend(texts)
        return np.zeros((len(texts), 1, 2), dtype=np.float32)

    model.extract = fake_extract
    model.extract_records([_record()])
    assert captured == ["Q: Who wrote Hamlet? A:"]


def test_trajectory_concatenates_prompt_checkpoint_and_final_deltas():
    record = {
        "prompt": "Q: Count? A:",
        "answer": "one two three four five",
    }
    model = HFActivationModel.__new__(HFActivationModel)
    model.config = ModelConfig(
        model_name="unused", activation_mode="trajectory", trajectory_checkpoints=(1, 4)
    )
    model.trajectory_checkpoints = (1, 4)

    class FakeTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return text.split()

        @staticmethod
        def decode(token_ids, **_kwargs):
            return " ".join(token_ids)

    model.tokenizer = FakeTokenizer()

    def fake_extract(texts):
        # State value equals the number of answer words observed.
        values = []
        for text in texts:
            suffix = text.split("A:", maxsplit=1)[1]
            observed = len(suffix.split())
            values.append(np.full((1, 2), observed, dtype=np.float32))
        return np.stack(values, axis=0)

    model.extract = fake_extract
    result = model.extract_records([record])
    assert result.shape == (1, 1, 8)
    np.testing.assert_allclose(result[0, 0], [0, 0, 1, 1, 4, 4, 5, 5])
