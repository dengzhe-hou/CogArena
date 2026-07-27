from scripts.posthoc.signature_family_sensitivity import (
    exact_sign_flip,
    summarize_family_means,
)


def test_family_means_equal_weight_families_not_checkpoints():
    result = summarize_family_means(
        {"a1": 1.0, "a2": 1.0, "a3": 1.0, "b1": -1.0},
        {"a1": "a", "a2": "a", "a3": "a", "b1": "b"},
    )
    assert result["n_families"] == 2
    assert result["fraction"] == "1/2"
    assert result["mean_of_family_mean_deltas"] == 0.0


def test_exact_sign_flip_enumerates_all_assignments():
    result = exact_sign_flip([1.0, 1.0])
    assert result["n_permutations"] == 4
    assert result["observed_mean_delta"] == 1.0
    assert result["p_one_sided"] == 0.25
    assert result["p_two_sided"] == 0.5
