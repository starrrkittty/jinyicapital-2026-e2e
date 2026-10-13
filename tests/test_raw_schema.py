"""Raw-field schema profiles and the A-side adapter.

The platform field dictionary (42 columns, five book levels, ``pre_close``) and the
downloaded package (28 columns, three book levels, no ``pre_close``) disagree, so both
profiles are encoded. These cases pin **which one is active** and prove the adapter fails
loudly instead of silently dropping columns. Only numpy/pandas/pyarrow are needed.
"""

import numpy as np
import pandas as pd
import pytest

from ptop import schema
from ptop.data import normalize_bars
from ptop.schema import FEATURES, SCHEMA_VERSION, ContractError

# The 25-field layout the downloaded Feather packages actually carry.
BOOK3_FIELDS = (
    "open", "high", "low", "close", "volume", "amount", "deal_number",
    "bid_price1", "ask_price1", "bid_price2", "ask_price2", "bid_price3", "ask_price3",
    "bid_volume1", "ask_volume1", "bid_volume2", "ask_volume2", "bid_volume3", "ask_volume3",
    "bid_num_orders1", "ask_num_orders1", "bid_num_orders2", "ask_num_orders2",
    "bid_num_orders3", "ask_num_orders3",
)

# The platform dictionary: 38 numeric columns, grouped as K-line + pre_close + trade
# aggregates + book (price / volume / order counts) for levels 1..5.
BOOK5_FIELDS = (
    ("open", "high", "low", "close", "pre_close", "volume", "amount", "deal_number")
    + tuple(f"{side}_price{level}" for level in range(1, 6) for side in ("bid", "ask"))
    + tuple(f"{side}_volume{level}" for level in range(1, 6) for side in ("bid", "ask"))
    + tuple(f"{side}_num_orders{level}" for level in range(1, 6) for side in ("bid", "ask"))
)


def test_active_profile_is_the_downloaded_package():
    """Resolved by measuring the real file: 28 columns, three levels, no pre_close."""
    assert SCHEMA_VERSION == "ptop.raw25.v1"
    assert FEATURES == BOOK3_FIELDS
    assert len(FEATURES) == 25
    assert len(FEATURES) == len(set(FEATURES))


def test_both_profiles_remain_encoded_and_differ_by_thirteen_fields():
    assert schema.FEATURES_BOOK3 == BOOK3_FIELDS
    assert schema.FEATURES_BOOK5 == BOOK5_FIELDS
    assert len(schema.FEATURES_BOOK5) == 38
    extra = set(schema.FEATURES_BOOK5) - set(schema.FEATURES_BOOK3)
    expected_extra = {"pre_close"} | {
        f"{kind}{level}"
        for level in (4, 5)
        for kind in ("ask_price", "bid_price", "ask_volume", "bid_volume", "ask_num_orders", "bid_num_orders")
    }
    assert extra == expected_extra, "5 档比 3 档多的应恰好是 pre_close + 第 4/5 档的两侧价量笔数"
    assert not set(schema.FEATURES_BOOK3) - set(schema.FEATURES_BOOK5), "3 档字段应全部包含在 5 档里"


def test_derived_tuples_stay_inside_the_active_profile():
    for name in ("BOOK_PRICE_FIELDS", "BOOK_VOLUME_FIELDS", "BOOK_ORDER_FIELDS",
                 "BOOK_FIELDS", "PRICE_FIELDS", "SCALE_FIELDS", "BOOK12_FIELDS"):
        members = getattr(schema, name)
        assert set(members) <= set(FEATURES), f"{name} holds fields outside FEATURES"
        assert len(members) == len(set(members)), f"{name} has duplicates"
    assert len(schema.BOOK12_FIELDS) == 12
    # 3 档 profile 下 pre_close 不存在，价格类字段不能引用它
    assert "pre_close" not in schema.PRICE_FIELDS
    assert set(schema.PRICE_FIELDS) == set(schema.OHLC_FIELDS) | set(schema.BOOK_PRICE_FIELDS)


def _frame(columns):
    rows = []
    for stamp in ("2024-01-02 10:00:00", "2024-01-02 10:30:00"):
        for instrument_id, name in ((1, "000001.SZ"), (2, "000002.SZ")):
            row = {"date": stamp, "instrument_id": instrument_id, "instrument": name}
            for position, field in enumerate(columns):
                row[field] = 1.0 + 0.01 * position + 0.001 * instrument_id
            rows.append(row)
    return pd.DataFrame(rows)


def test_adapter_accepts_the_active_profile_table():
    canonical = normalize_bars(_frame(FEATURES), source="cloud",
                               instrument_map={1: "000001.SZ", 2: "000002.SZ"})
    assert tuple(canonical.values.columns) == FEATURES
    assert canonical.values.index.names == ["datetime", "instrument"]
    assert canonical.observed.to_numpy().all()
    assert canonical.metadata["instrument"].tolist() == ["000001.SZ", "000002.SZ"] * 2


def test_adapter_selects_the_active_fields_from_a_wider_table():
    """A 5-level table is accepted but only FEATURES is kept — an explicit selection, not a guess."""
    canonical = normalize_bars(_frame(BOOK5_FIELDS), source="cloud",
                               instrument_map={1: "000001.SZ", 2: "000002.SZ"})
    assert tuple(canonical.values.columns) == FEATURES
    assert "pre_close" not in canonical.values.columns


def test_adapter_rejects_a_table_missing_the_active_book_fields():
    """Dropping level-3 fields must name exactly what is missing."""
    partial = [field for field in FEATURES if not field.endswith("3")]
    with pytest.raises(ContractError) as error:
        normalize_bars(_frame(partial), source="cloud", instrument_map={1: "000001.SZ", 2: "000002.SZ"})
    message = str(error.value)
    assert "缺少必要字段" in message
    for field in ("bid_price3", "ask_volume3", "bid_num_orders3"):
        assert f"'{field}'" in message
    for field in ("open", "deal_number", "ask_price1"):
        assert f"'{field}'" not in message, "不得把已经存在的字段报成缺失"


def test_local_source_scales_price_fields_but_not_volume():
    """Every price-like field shares the price unit contract; volume must stay raw."""
    frame = _frame(FEATURES)
    frame = frame.assign(**{field: 1000 + position for position, field in enumerate(FEATURES)})
    frame = frame.astype({field: "int64" for field in FEATURES})
    local = normalize_bars(frame, source="local", instrument_map={1: "000001.SZ", 2: "000002.SZ"})
    for field in ("close", "bid_price3", "ask_price1"):
        assert field in schema.PRICE_FIELDS
        np.testing.assert_allclose(local.values[field].to_numpy(), frame[field].to_numpy() / 100.0)
    np.testing.assert_allclose(local.values["amount"].to_numpy(), frame["amount"].to_numpy() / 100.0)
    np.testing.assert_allclose(local.values["volume"].to_numpy(), frame["volume"].to_numpy())
