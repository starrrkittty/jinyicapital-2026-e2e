"""Canonical raw-field schema for the 2026 Jinyi Capital end-to-end track.

Field names and order are the contract shared by A's adapter, the frozen
preprocessor, every encoder and the saved weights. Every entry is a **raw platform
column**. No hand-made indicator, ratio, price-volume combination, book-derived
statistic, factor expression or screening rule is admitted, so the model only ever
sees raw quotes and book state (see the track rules and docs/objective_training_plan.md).

Two facts about the official table are load-bearing:

- ``date`` is the bar's **end** time, so a bar is fully available as soon as it is
  named; callers must pass ``bar_end_is_available=True`` rather than have it assumed.
- ``adjust_factor`` is metadata for audit/alignment, not a model input. Windows never
  cross days, so unadjusted intraday price levels are self-consistent.

Schema profile
--------------
The platform field dictionary and A's scanned download disagree, so both profiles are
encoded and the difference is two constants (``_LEVELS``, ``_WITH_REFERENCE``):

- ``FEATURES_BOOK5`` (38 fields) — the platform dictionary: K-line, ``pre_close``,
  trade aggregates and **five** book levels of price / volume / order counts.
- ``FEATURES_BOOK3`` (25 fields) — what A actually observed in the downloaded
  ``cpt_jyc_2026_e2e_bar*.tar.gz`` packages: three book levels and no ``pre_close``
  (docs/b_data_reference.md). This is the historical ``ptop.raw25.v1`` layout.

``FEATURES`` and every derived tuple come from the active profile, so switching is one
edit plus a ``SCHEMA_VERSION`` bump. Switching changes the tensor width and therefore
invalidates saved weights.

**Active profile: ``FEATURES_BOOK3`` (``ptop.raw25.v1``).** This was resolved by measuring
the real package rather than by preference: ``cpt_jyc_2026_e2e_bar30m/202001.0.feather`` has
exactly 28 columns (three book levels, no ``pre_close``), which is ``FEATURES_BOOK3`` plus
``date`` / ``instrument_id`` / ``adjust_factor``. The 5-level profile remains encoded and
tested for the case where a 5-level table is used instead; do not mix the two within one
training run or one set of weights.
"""

QLIB_COMMIT = "54355232463878d2eebb91fe0ee5fa7fa1f5976c"

OHLC_FIELDS = ("open", "high", "low", "close")
# 官方原始列；窗口不跨日，pre_close 是当日唯一可得的隔夜价格参照。
REFERENCE_FIELDS = ("pre_close",)
TRADE_FIELDS = ("volume", "amount", "deal_number")


def _book_kind(kind: str, levels: int) -> tuple[str, ...]:
    if type(levels) is not int or levels < 1:
        raise ValueError("book level count must be a positive integer")
    return tuple(f"{side}_{kind}{level}" for level in range(1, levels + 1) for side in ("bid", "ask"))


def book_fields(levels: int) -> tuple[str, ...]:
    """Price, then volume, then order-count fields for ``levels`` book levels."""
    return _book_kind("price", levels) + _book_kind("volume", levels) + _book_kind("num_orders", levels)


def features_for(levels: int, *, with_reference: bool = True) -> tuple[str, ...]:
    """Compose one raw-field profile: K-line + optional reference + trade + book levels."""
    reference = REFERENCE_FIELDS if with_reference else ()
    return OHLC_FIELDS + reference + TRADE_FIELDS + book_fields(levels)


# --- profiles -----------------------------------------------------------------
FEATURES_BOOK5 = features_for(5)
FEATURES_BOOK3 = features_for(3, with_reference=False)

# Active profile. Measured on the actual downloaded package
# (``cpt_jyc_2026_e2e_bar30m/202001.0.feather``: 28 columns, three book levels, no
# ``pre_close``), so the download and the model agree at 25 fields. Inference only
# ``SELECT``s the columns named in ``FEATURES`` and the platform tables are a superset of
# them, so training and inference stay the same width. Switching is this line plus a
# ``SCHEMA_VERSION`` bump; changing the width invalidates saved weights.
_LEVELS, _WITH_REFERENCE = 3, False
FEATURES = features_for(_LEVELS, with_reference=_WITH_REFERENCE)
SCHEMA_VERSION = "ptop.raw25.v1"

BOOK_PRICE_FIELDS = _book_kind("price", _LEVELS)
BOOK_VOLUME_FIELDS = _book_kind("volume", _LEVELS)
BOOK_ORDER_FIELDS = _book_kind("num_orders", _LEVELS)
BOOK_FIELDS = BOOK_PRICE_FIELDS + BOOK_VOLUME_FIELDS + BOOK_ORDER_FIELDS
# 价格类字段共享同一价格尺度（N2 共用归一化），缺失组仍单独保留。
PRICE_FIELDS = OHLC_FIELDS + (REFERENCE_FIELDS if _WITH_REFERENCE else ()) + BOOK_PRICE_FIELDS
SCALE_FIELDS = PRICE_FIELDS + ("amount",)

# 配对 DeepLOB 复现固定使用 1-3 档的两价两量四字段；不随档位数变化。
DEEPBOOK_LEVELS = 3
BOOK12_FIELDS = tuple(
    f"{field}{level}"
    for level in range(1, DEEPBOOK_LEVELS + 1)
    for field in ("ask_price", "ask_volume", "bid_price", "bid_volume")
)


class ContractError(ValueError):
    pass


# 尚未向官方确认的口径会用这些词占位。占位值必须让流水线停下来，而不是悄悄进入训练：
# 「预测目标」的残差化口径直接决定 head 拟合什么，猜错等于整轮训练作废。
CONTRACT_PLACEHOLDER_TOKENS = (
    "tbd", "todo", "placeholder", "fixme", "pending", "待确认", "待定", "未定", "待官方", "待填", "unknown", "xxx",
)


def reject_placeholder(name: str, value: object) -> str:
    """Return a confirmed contract string, or raise if it is empty or still a placeholder."""
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} 不能为空")
    stripped = value.strip()
    lowered = stripped.lower()
    if any(token in lowered for token in CONTRACT_PLACEHOLDER_TOKENS) or set(lowered) <= {"?", "-", "."}:
        raise ContractError(
            f"{name} 仍是占位值（{value!r}）。请先向官方确认口径，不要用占位值启动训练或导出。"
        )
    return stripped
