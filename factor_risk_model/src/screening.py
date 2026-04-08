"""Equity factor screening and ML ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_12M = 252

FEATURE_COLUMNS = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12_1",
    "volatility_3m",
    "volatility_6m",
    "drawdown_6m",
    "market_relative_3m",
    "market_relative_6m",
    "short_term_reversal",
]


@dataclass
class ScreeningResult:
    """Artifacts for the current screen and classifier."""

    latest_screen: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_metrics: Dict[str, float] = field(default_factory=dict)
    model_coefficients: pd.Series = field(default_factory=pd.Series)
    holdout_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _safe_zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    if not higher_is_better:
        clean = -clean
    std = clean.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((clean - clean.mean()) / std).fillna(0.0)


def build_factor_screen(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Rank securities by value, momentum, and quality."""

    if prices.empty or fundamentals.empty:
        raise ValueError("prices and fundamentals must be non-empty")
    if len(prices) <= LOOKBACK_12M:
        raise ValueError("prices must contain at least 252 observations")

    price_frame = prices.ffill().dropna(axis=1, how="all")
    fundamentals = fundamentals.reindex(price_frame.columns)

    momentum_12_1 = price_frame.shift(LOOKBACK_1M).pct_change(
        LOOKBACK_12M - LOOKBACK_1M
    ).iloc[-1]
    momentum_6_1 = price_frame.shift(LOOKBACK_1M).pct_change(
        LOOKBACK_6M - LOOKBACK_1M
    ).iloc[-1]
    latest_price = price_frame.iloc[-1]
    trailing_return_1m = latest_price / price_frame.iloc[-LOOKBACK_1M - 1] - 1.0

    value_score = pd.concat(
        [
            _safe_zscore(fundamentals["trailingPE"], higher_is_better=False),
            _safe_zscore(fundamentals["forwardPE"], higher_is_better=False),
            _safe_zscore(fundamentals["priceToBook"], higher_is_better=False),
            _safe_zscore(fundamentals["enterpriseToEbitda"], higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1)

    quality_score = pd.concat(
        [
            _safe_zscore(fundamentals["returnOnEquity"]),
            _safe_zscore(fundamentals["returnOnAssets"]),
            _safe_zscore(fundamentals["profitMargins"]),
            _safe_zscore(fundamentals["operatingMargins"]),
            _safe_zscore(fundamentals["currentRatio"]),
            _safe_zscore(fundamentals["debtToEquity"], higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1)

    momentum_score = pd.concat(
        [
            _safe_zscore(momentum_12_1),
            _safe_zscore(momentum_6_1),
            _safe_zscore(trailing_return_1m),
        ],
        axis=1,
    ).mean(axis=1)

    screen = pd.DataFrame(index=price_frame.columns)
    screen["name"] = fundamentals["shortName"]
    screen["sector"] = fundamentals["sector"]
    screen["trailing_pe"] = fundamentals["trailingPE"]
    screen["forward_pe"] = fundamentals["forwardPE"]
    screen["price_to_book"] = fundamentals["priceToBook"]
    screen["return_on_equity"] = fundamentals["returnOnEquity"]
    screen["profit_margin"] = fundamentals["profitMargins"]
    screen["debt_to_equity"] = fundamentals["debtToEquity"]
    screen["momentum_12_1"] = momentum_12_1
    screen["momentum_6_1"] = momentum_6_1
    screen["trailing_return_1m"] = trailing_return_1m
    screen["value_score"] = value_score
    screen["momentum_score"] = momentum_score
    screen["quality_score"] = quality_score
    screen["factor_score"] = (
        0.35 * screen["value_score"]
        + 0.35 * screen["momentum_score"]
        + 0.30 * screen["quality_score"]
    )
    screen["factor_rank"] = (
        screen["factor_score"].rank(ascending=False, method="dense").astype(int)
    )
    return screen.sort_values(["factor_score", "momentum_12_1"], ascending=[False, False])


def _feature_block(prices: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    """Build cross-sectional features for one rebalance date."""

    returns = prices.pct_change()
    latest = prices.iloc[row_idx]

    block = pd.DataFrame(index=prices.columns)
    block["momentum_1m"] = latest / prices.iloc[row_idx - LOOKBACK_1M] - 1.0
    block["momentum_3m"] = latest / prices.iloc[row_idx - LOOKBACK_3M] - 1.0
    block["momentum_6m"] = latest / prices.iloc[row_idx - LOOKBACK_6M] - 1.0
    block["momentum_12_1"] = (
        prices.iloc[row_idx - LOOKBACK_1M] / prices.iloc[row_idx - LOOKBACK_12M] - 1.0
    )
    block["volatility_3m"] = (
        returns.iloc[row_idx - LOOKBACK_3M + 1 : row_idx + 1].std() * np.sqrt(252)
    )
    block["volatility_6m"] = (
        returns.iloc[row_idx - LOOKBACK_6M + 1 : row_idx + 1].std() * np.sqrt(252)
    )
    block["drawdown_6m"] = (
        latest / prices.iloc[row_idx - LOOKBACK_6M : row_idx + 1].max() - 1.0
    )
    block["market_relative_3m"] = (
        block["momentum_3m"] - block["momentum_3m"].mean()
    )
    block["market_relative_6m"] = (
        block["momentum_6m"] - block["momentum_6m"].mean()
    )
    block["short_term_reversal"] = -block["momentum_1m"]
    return block


def build_classification_dataset(
    prices: pd.DataFrame,
    rebalance_frequency: int = LOOKBACK_1M,
    lookahead: int = LOOKBACK_1M,
) -> pd.DataFrame:
    """Create a cross-sectional dataset for return classification."""

    if prices.empty:
        raise ValueError("prices must be non-empty")
    if len(prices) <= LOOKBACK_12M + lookahead:
        raise ValueError("prices must contain sufficient history")

    prices = prices.ffill().dropna(axis=1, how="all")
    rows: List[Dict[str, float]] = []

    for row_idx in range(LOOKBACK_12M, len(prices) - lookahead, rebalance_frequency):
        block = _feature_block(prices, row_idx)
        forward_return = prices.iloc[row_idx + lookahead] / prices.iloc[row_idx] - 1.0
        forward_excess = forward_return - forward_return.mean()
        targets = (forward_excess > 0).astype(int)

        block["date"] = prices.index[row_idx]
        block["symbol"] = block.index
        block["target"] = targets
        rows.append(block.reset_index(drop=True))

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["target"])
    return dataset


def fit_return_classifier(
    dataset: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> Tuple[Pipeline, Dict[str, float], pd.Series, pd.DataFrame]:
    """Fit a time-split logistic classifier for cross-sectional outperformance."""

    features = list(feature_columns)
    if dataset.empty:
        raise ValueError("dataset must be non-empty")

    unique_dates = sorted(pd.to_datetime(dataset["date"]).unique())
    if len(unique_dates) < 8:
        raise ValueError("need at least 8 rebalance dates to train classifier")

    split_idx = max(1, int(len(unique_dates) * 0.75))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    train = dataset[dataset["date"].isin(train_dates)].copy()
    test = dataset[dataset["date"].isin(test_dates)].copy()

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(train[features], train["target"])

    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "holdout_accuracy": float(accuracy_score(test["target"], predictions)),
        "holdout_precision": float(
            precision_score(test["target"], predictions, zero_division=0)
        ),
        "holdout_recall": float(
            recall_score(test["target"], predictions, zero_division=0)
        ),
        "holdout_roc_auc": float(roc_auc_score(test["target"], probabilities)),
        "holdout_brier_score": float(brier_score_loss(test["target"], probabilities)),
        "holdout_start": str(pd.Timestamp(test_dates[0]).date()),
        "holdout_end": str(pd.Timestamp(test_dates[-1]).date()),
    }

    coefficient_series = pd.Series(
        model.named_steps["classifier"].coef_[0],
        index=features,
        name="coefficient",
    ).sort_values(key=np.abs, ascending=False)

    holdout_predictions = test[["date", "symbol", "target"]].copy()
    holdout_predictions["predicted_probability"] = probabilities
    holdout_predictions["predicted_class"] = predictions

    return model, metrics, coefficient_series, holdout_predictions


def _latest_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if len(prices) <= LOOKBACK_12M:
        raise ValueError("prices must contain at least 252 observations")
    return _feature_block(prices.ffill(), len(prices) - 1)


def run_equity_screening(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    rebalance_frequency: int = LOOKBACK_1M,
    lookahead: int = LOOKBACK_1M,
) -> ScreeningResult:
    """Run the current factor screen and ML classifier."""

    screen = build_factor_screen(prices, fundamentals)
    dataset = build_classification_dataset(
        prices,
        rebalance_frequency=rebalance_frequency,
        lookahead=lookahead,
    )
    model, metrics, coefficients, holdout = fit_return_classifier(dataset)

    latest_features = _latest_feature_frame(prices).reindex(screen.index)
    probabilities = model.predict_proba(latest_features[FEATURE_COLUMNS])[:, 1]
    screen["predicted_outperformance_probability"] = probabilities
    probability_score = _safe_zscore(screen["predicted_outperformance_probability"])
    screen["screen_score"] = 0.70 * screen["factor_score"] + 0.30 * probability_score
    screen["screen_rank"] = (
        screen["screen_score"].rank(ascending=False, method="dense").astype(int)
    )
    screen = screen.sort_values(
        ["screen_score", "predicted_outperformance_probability"],
        ascending=[False, False],
    )

    return ScreeningResult(
        latest_screen=screen,
        model_metrics=metrics,
        model_coefficients=coefficients,
        holdout_predictions=holdout.sort_values(["date", "predicted_probability"], ascending=[True, False]),
    )
