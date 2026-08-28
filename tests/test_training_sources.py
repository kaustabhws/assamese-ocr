import pytest

from axomiya_ocr.training.engine import _mixed_sampler, _source_weights


def test_hard_character_weights_preserve_requested_source_mass() -> None:
    weights = _source_weights(["অসম", "ভাষা", "সাহিত্য"], 0.9, {"ষ"}, 2.0)
    assert sum(weights) == pytest.approx(0.9)
    assert weights[1] == pytest.approx(weights[0] * 2.0)


def test_mixed_sampler_keeps_real_epoch_size() -> None:
    sampler = _mixed_sampler(["অসম", "ভাষা"], ["সাহিত্য"], 0.2, {"ষ"}, 2.0)
    assert sampler.num_samples == 2


def test_mixed_sampler_rejects_invalid_boost() -> None:
    with pytest.raises(ValueError, match="at least 1.0"):
        _mixed_sampler(["অসম"], [], 0.0, {"ষ"}, 0.5)

