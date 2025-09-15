import pandas as pd

from vectr_lab.risk.sizer import sizes_fixed_risk


def test_sizes_fixed_risk_basic():
    prices = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22]})
    sl = prices - 1
    sizes = sizes_fixed_risk(prices, sl, risk_pct=0.01)
    expected_a = 0.01 / 1
    expected_b = 0.01 / 1
    assert sizes.loc[0, "A"] == expected_a
    assert sizes.loc[0, "B"] == expected_b


def test_sizes_fixed_risk_zero_when_invalid():
    prices = pd.DataFrame({"A": [10, 11], "B": [10, 11]})
    sl = pd.DataFrame({"A": [10.5, 10.5], "B": [9.5, 9.5]})
    sizes = sizes_fixed_risk(prices, sl, risk_pct=0.01)
    assert sizes.loc[0, "A"] == 0.0
    assert sizes.loc[0, "B"] > 0
