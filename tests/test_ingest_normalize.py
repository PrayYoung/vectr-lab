import pandas as pd

from vectr_lab.data.ingest import _normalize_ohlcv


def _multiindex_frame(ticker: str) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            (ticker, "Open", "RegularMarket"),
            (ticker, "High", "RegularMarket"),
            (ticker, "Low", "RegularMarket"),
            (ticker, "Close", "RegularMarket"),
            (ticker, "Volume", "RegularMarket"),
        ]
    )
    data = {
        (ticker, "Open", "RegularMarket"): [1.0, 1.1, 1.2],
        (ticker, "High", "RegularMarket"): [1.2, 1.3, 1.4],
        (ticker, "Low", "RegularMarket"): [0.9, 1.0, 1.1],
        (ticker, "Close", "RegularMarket"): [1.05, 1.15, 1.25],
        (ticker, "Volume", "RegularMarket"): [100, 110, 120],
    }
    df = pd.DataFrame(data, index=index)
    return df


def test_normalize_multiindex_to_flat_frame():
    df = _multiindex_frame("AAPL")
    normalized = _normalize_ohlcv(df, "AAPL")
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.name == "timestamp"
    assert not isinstance(normalized.columns, pd.MultiIndex)


def test_normalize_already_flat_frame():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 110],
        },
        index=index,
    )
    normalized = _normalize_ohlcv(df, "AAPL")
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_index_equal(normalized.index, index.rename("timestamp"))
    pd.testing.assert_frame_equal(normalized, df.set_index(index).rename_axis("timestamp"))
