import pytest

from dubber import config
from dubber.speakers import decide_gender

T = config.F0_MALE_MAX_HZ


def test_agree_male():
    g, c = decide_gender(T - 60, 0.95)
    assert g == "male" and c == pytest.approx(0.6 + 0.4 * 0.9)


def test_agree_female():
    g, c = decide_gender(T + 75, 0.0)
    assert g == "female" and c == pytest.approx(1.0)


def test_disagree_follows_f0():
    assert decide_gender(T - 40, 0.1) == ("male", 0.4)


def test_f0_only():
    g, c = decide_gender(T + 30, None)
    assert g == "female" and c == pytest.approx(min(0.55, 0.3 + 0.3 * 0.5))
    assert decide_gender(T - 200, None)[1] == pytest.approx(0.55)


def test_no_f0():
    assert decide_gender(None, 0.9) == ("unknown", 0.0)


def test_low_confidence_threshold():
    # F0-only and disagreement can never reach GENDER_CONF_MIN; a weak agreement can sit just below it
    assert decide_gender(T - 200, None)[1] < config.GENDER_CONF_MIN
    assert decide_gender(T - 40, 0.1)[1] < config.GENDER_CONF_MIN
    assert decide_gender(T - 60, 0.5001)[1] >= config.GENDER_CONF_MIN  # agree => 0.6 + ..., never below
