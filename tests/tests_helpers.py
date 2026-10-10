def assert_prediction_list(preds):
    assert isinstance(preds, list)
    for p in preds:
        assert set(p) == {"name", "confidence"}
        assert isinstance(p["name"], str)
        assert 0.0 <= p["confidence"] <= 1.0
