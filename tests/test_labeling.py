from haloscope.labeling import RougeLScorer, label_records, rouge_l_f1


def test_rouge_l():
    assert rouge_l_f1("Paris", "Paris") == 1.0
    assert rouge_l_f1("completely wrong", "Paris") == 0.0
    assert 0.0 < rouge_l_f1("the city of Paris", "Paris") < 1.0


def test_label_records_uses_best_reference():
    records = [{"answer": "Paris", "references": ["London", "Paris"], "id": "x"}]
    labeled = label_records(records, RougeLScorer(), threshold=0.5)
    assert labeled[0]["truth_label"] == 1
    assert labeled[0]["similarity"] == 1.0

