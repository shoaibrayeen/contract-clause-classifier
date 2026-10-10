from tests_helpers import assert_prediction_list


def test_single_provision(predictor):
    out = predictor.predict_document("This Agreement shall be governed by the laws of Delaware.", segment=False)
    assert out["num_clauses"] == 1
    assert out["clauses"][0]["predictions"][0]["name"] == "Governing Laws"
    assert_prediction_list(out["document"])


def test_document_aggregation(predictor):
    text = (
        "1. Confidentiality. Each party shall keep all confidential information secret and protect it with reasonable care at all times.\n\n"
        "2. Termination. Either party may terminate this agreement on thirty days written notice to the other party for any reason."
    )
    out = predictor.predict_document(text)
    names = [d["name"] for d in out["document"]]
    assert "Confidentiality" in names and "Terminations" in names
    assert out["num_clauses"] == 2
    confs = [d["confidence"] for d in out["document"]]
    assert confs == sorted(confs, reverse=True)
    for c in out["clauses"]:
        assert_prediction_list(c["predictions"])


def test_top1_always_kept_even_below_threshold(predictor):
    out = predictor.predict_document("Nothing relevant here at all.", segment=False, threshold=0.99)
    assert len(out["clauses"][0]["predictions"]) == 1


def test_empty_text(predictor):
    assert predictor.predict_document("") == {"document": [], "clauses": [], "num_clauses": 0}
