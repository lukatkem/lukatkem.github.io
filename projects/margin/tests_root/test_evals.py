"""Eval metric math + answer checking."""
from margin.evals import check_answer, load_golden
from margin.config import GOLDEN_PATH


def test_check_answer_case_insensitive_all_terms():
    assert check_answer("Max daily loss is 5% of equity.", ["5%", "equity"])
    assert not check_answer("Max daily loss is 5% of equity.", ["5%", "balance"])
    assert check_answer("RESET AT 00:00 UTC", ["00:00 utc"])  # case


def test_golden_dataset_loads_and_is_well_formed():
    golden = load_golden(GOLDEN_PATH)
    assert len(golden) >= 60, "expect a substantial golden set"
    ids = [q["id"] for q in golden]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in golden:
        assert q["question"] and q["gold_doc"] and q["must_contain"]
