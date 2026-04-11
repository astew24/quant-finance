from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "assets" / "data" / "portfolio.json"
DOCS_LIVE = ROOT / "docs" / "live.html"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def cumulative_curve(returns: list[float], base: float = 0.0) -> list[float]:
    level = 1.0 + base
    curve = []
    for ret in returns:
        level *= 1.0 + ret
        curve.append(level - 1.0)
    return curve


def normalized_price_index(returns: list[float], base: float = 100.0) -> list[float]:
    level = base
    prices = []
    for ret in returns:
        level *= math.exp(ret)
        prices.append(level)
    return prices


def downsample(rows: list[dict[str, float | str | None]], max_points: int) -> list[dict[str, float | str | None]]:
    if len(rows) <= max_points:
        return rows
    step = max(1, math.ceil(len(rows) / max_points))
    sampled = rows[::step]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled


def prettify_factor(name: str) -> str:
    return name.replace("_", " ").title().replace("12 1", "12-1")


def parse_crypto_report(text: str) -> dict[str, dict[str, float]]:
    pattern = re.compile(
        r"--- (?P<symbol>[A-Z\-]+) ---.*?"
        r"Walk-forward backtest:\s+RMSE\s+:\s+(?P<rmse>[0-9.]+).*?"
        r"vs Random Walk\s+RMSE=(?P<random_walk>[0-9.]+)",
        flags=re.S,
    )
    baselines: dict[str, dict[str, float]] = {}
    for match in pattern.finditer(text):
        baselines[match.group("symbol")] = {
            "walk_forward_rmse": float(match.group("rmse")),
            "random_walk_rmse": float(match.group("random_walk")),
        }
    return baselines


def parse_quantamental_ideas(markdown_text: str) -> list[dict[str, str]]:
    sections = re.split(r"^## ", markdown_text, flags=re.M)[1:]
    ideas: list[dict[str, str]] = []
    for section in sections:
        lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
        ticker = lines[0]
        thesis = ""
        risk = ""
        for line in lines[1:]:
            if line.startswith("- Investment thesis:"):
                thesis = line.replace("- Investment thesis:", "").strip()
            if line.startswith("- Key risks:"):
                risk = line.replace("- Key risks:", "").strip()
        ideas.append({"ticker": ticker, "thesis": thesis, "risk": risk})
    return ideas


def parse_options_output(text: str) -> dict[str, float]:
    patterns = {
        "black_scholes_price": r"Black-Scholes price\s+:\s+([0-9.]+)",
        "implied_volatility_pct": r"Implied volatility\s+:\s+([0-9.]+)%",
        "monte_carlo_price": r"Monte Carlo price\s+:\s+([0-9.]+)",
        "monte_carlo_error": r"Monte Carlo price\s+:\s+[0-9.]+ \+/- ([0-9.]+)",
        "american_tree_price": r"American tree price\s+:\s+([0-9.]+)",
        "delta": r"delta\s+([0-9.\-]+)",
        "gamma": r"gamma\s+([0-9.\-]+)",
        "vega": r"vega\s+([0-9.\-]+)",
        "theta": r"theta\s+([0-9.\-]+)",
        "rho": r"rho\s+([0-9.\-]+)",
    }
    output: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            output[key] = float(match.group(1))
    return output


def build_crypto_payload() -> dict[str, object]:
    summary_rows = read_csv_rows(ROOT / "crypto_volatility" / "output_sample" / "summary.csv")
    report_text = (ROOT / "crypto_volatility" / "output_sample" / "report.txt").read_text(encoding="utf-8")
    baselines = parse_crypto_report(report_text)

    assets = []
    for summary in summary_rows:
        symbol = summary["symbol"]
        file_stem = symbol.replace("-", "_")
        timeseries = read_csv_rows(ROOT / "crypto_volatility" / "output_sample" / f"{file_stem}_timeseries.csv")
        forecast = read_csv_rows(ROOT / "crypto_volatility" / "output_sample" / f"{file_stem}_forecast.csv")
        strategy = read_csv_rows(ROOT / "crypto_volatility" / "output_sample" / f"{file_stem}_strategy.csv")

        returns = [parse_float(row["returns"]) or 0.0 for row in timeseries]
        price_index = normalized_price_index(returns)
        price_rows = downsample(
            [
                {
                    "date": row["Date"],
                    "price_index": round(price, 3),
                }
                for row, price in zip(timeseries, price_index)
            ],
            max_points=110,
        )

        vol_rows = downsample(
            [
                {
                    "date": row["Date"],
                    "realized_vol": parse_float(row["realised_vol"]),
                    "conditional_vol": parse_float(row["conditional_vol"]),
                }
                for row in timeseries
            ],
            max_points=110,
        )

        returns_by_date = {row["Date"]: parse_float(row["returns"]) or 0.0 for row in timeseries}
        strategy_returns = [parse_float(row["strategy_returns"]) or 0.0 for row in strategy]
        strategy_curve = cumulative_curve(strategy_returns)
        benchmark_curve = cumulative_curve([returns_by_date[row[""]] for row in strategy])

        overlay_rows = downsample(
            [
                {
                    "date": row[""],
                    "strategy_curve": round(strategy_value, 4),
                    "benchmark_curve": round(benchmark_value, 4),
                    "position": round(parse_float(row["position"]) or 0.0, 4),
                }
                for row, strategy_value, benchmark_value in zip(strategy, strategy_curve, benchmark_curve)
            ],
            max_points=95,
        )

        forecast_rows = [
            {
                "date": row[""],
                "forecast_vol": round(parse_float(row["forecast_vol"]) or 0.0, 6),
            }
            for row in forecast
        ]

        assets.append(
            {
                "symbol": symbol,
                "label": symbol.replace("-USD", ""),
                "date_range": {
                    "start": timeseries[0]["Date"],
                    "end": timeseries[-1]["Date"],
                },
                "metrics": {
                    "annual_volatility": parse_float(summary["ann_vol"]),
                    "var_95": parse_float(summary["VaR_95"]),
                    "walk_forward_rmse": parse_float(summary["garch_backtest_rmse"]),
                    "direction_accuracy": parse_float(summary["garch_backtest_direction_accuracy"]),
                    "strategy_sharpe": parse_float(summary["strategy_sharpe_ratio"]),
                    "strategy_average_leverage": parse_float(summary["strategy_average_leverage"]),
                    "strategy_max_drawdown": parse_float(summary["strategy_max_drawdown"]),
                    "forecast_end": parse_float(forecast[-1]["forecast_vol"]),
                    "random_walk_rmse": baselines.get(symbol, {}).get("random_walk_rmse"),
                },
                "price_series": price_rows,
                "vol_series": vol_rows,
                "forecast_series": forecast_rows,
                "overlay_series": overlay_rows,
            }
        )

    return {"assets": assets}


def build_factor_payload() -> dict[str, object]:
    summary = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "summary.csv")[0]
    metadata = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "run_metadata.csv")[0]
    screen_metrics = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "screening_model_metrics.csv")[0]
    strategy_rows = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "strategy_returns.csv")
    exposures_rows = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "factor_exposures.csv")
    latest_screen = read_csv_rows(ROOT / "factor_risk_model" / "output_sample" / "latest_screen.csv")
    ideas_md = (ROOT / "factor_risk_model" / "output_sample" / "top_quantamental_ideas.md").read_text(
        encoding="utf-8"
    )

    strategy_curve = cumulative_curve([parse_float(row["strategy_returns"]) or 0.0 for row in strategy_rows])
    benchmark_curve = cumulative_curve([parse_float(row["benchmark_returns"]) or 0.0 for row in strategy_rows])
    performance_rows = downsample(
        [
            {
                "date": row[""],
                "strategy_curve": round(strategy_value, 4),
                "benchmark_curve": round(benchmark_value, 4),
            }
            for row, strategy_value, benchmark_value in zip(strategy_rows, strategy_curve, benchmark_curve)
        ],
        max_points=120,
    )

    exposures = [
        {
            "factor": prettify_factor(row[""]),
            "beta": round(parse_float(row["beta"]) or 0.0, 4),
        }
        for row in exposures_rows
    ]

    top_screen = []
    for row in latest_screen[:5]:
        top_screen.append(
            {
                "ticker": row["Ticker"],
                "company": row["name"],
                "sector": row["sector"],
                "screen_rank": int(float(row["screen_rank"])),
                "screen_score": round(parse_float(row["screen_score"]) or 0.0, 3),
                "probability": round((parse_float(row["predicted_outperformance_probability"]) or 0.0) * 100.0, 1),
                "price": round(parse_float(row["currentPrice"]) or 0.0, 2),
                "target": round(parse_float(row["targetMeanPrice"]) or 0.0, 2),
            }
        )

    return {
        "metadata": {
            "start_date": metadata["start_date"],
            "end_date": metadata["end_date"],
            "backtest_universe_size": int(float(metadata["backtest_universe_size"])),
            "screening_universe_size": int(float(metadata["screening_universe_size"])),
            "transaction_cost_bps": int(float(metadata["transaction_cost_bps"])),
        },
        "metrics": {
            "total_return": parse_float(summary["total_return"]),
            "sharpe_ratio": parse_float(summary["sharpe_ratio"]),
            "mean_information_coefficient": parse_float(summary["mean_information_coefficient"]),
            "holdout_accuracy": parse_float(screen_metrics["holdout_accuracy"]),
            "holdout_roc_auc": parse_float(screen_metrics["holdout_roc_auc"]),
        },
        "performance_series": performance_rows,
        "factor_exposures": exposures,
        "top_screen": top_screen,
        "ideas": parse_quantamental_ideas(ideas_md)[:3],
    }


def build_options_payload() -> dict[str, object]:
    output = parse_options_output((ROOT / "options_pricing" / "output_sample.txt").read_text(encoding="utf-8"))
    return {
        "default_scenario": {
            "spot": 100.0,
            "strike": 100.0,
            "rate": 0.03,
            "volatility": 0.20,
            "maturity": 1.0,
            "option_type": "call",
        },
        "sample_output": output,
    }


def build_payload() -> dict[str, object]:
    crypto = build_crypto_payload()
    factor = build_factor_payload()
    options = build_options_payload()

    btc = next(asset for asset in crypto["assets"] if asset["symbol"] == "BTC-USD")
    factor_metrics = factor["metrics"]
    options_sample = options["sample_output"]

    return {
        "generated_for": "GitHub Pages portfolio microsite",
        "live_url": "https://astew24.github.io/quant-finance/",
        "repo_url": "https://github.com/astew24/quant-finance",
        "project_links": {
            "crypto": "https://github.com/astew24/quant-finance/tree/main/projects/crypto-volatility-risk-engine",
            "factor": "https://github.com/astew24/quant-finance/tree/main/projects/equity-factor-screening-pipeline",
            "options": "https://github.com/astew24/quant-finance/tree/main/projects/options-pricing-toolkit",
        },
        "hero_metrics": [
            {
                "label": "BTC Walk-Forward RMSE",
                "value": f"{btc['metrics']['walk_forward_rmse']:.3f}",
                "detail": f"vs random walk {btc['metrics']['random_walk_rmse']:.3f}",
            },
            {
                "label": "Factor Strategy Total Return",
                "value": f"{factor_metrics['total_return'] * 100.0:.1f}%",
                "detail": f"Sharpe {factor_metrics['sharpe_ratio']:.2f}",
            },
            {
                "label": "Options Cross-Check",
                "value": f"{options_sample['black_scholes_price']:.4f}",
                "detail": f"MC {options_sample['monte_carlo_price']:.4f} +/- {options_sample['monte_carlo_error']:.4f}",
            },
        ],
        "crypto": crypto,
        "factor": factor,
        "options": options,
    }


def build_live_html(payload: dict[str, object]) -> str:
    index_html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    styles = (ROOT / "docs" / "assets" / "css" / "styles.css").read_text(encoding="utf-8")
    script = (ROOT / "docs" / "assets" / "js" / "app.js").read_text(encoding="utf-8")
    payload_json = json.dumps(payload, indent=2).replace("</", "<\\/")

    html = index_html.replace(
        '<link rel="stylesheet" href="./assets/css/styles.css">',
        f"<style>\n{styles}\n</style>",
    )
    html = html.replace(
        '<script type="module" src="./assets/js/app.js"></script>',
        (
            '<script id="portfolio-data" type="application/json">\n'
            f"{payload_json}\n"
            "</script>\n"
            '<script type="module">\n'
            f"{script}\n"
            "</script>"
        ),
    )
    return html


def main() -> None:
    payload = build_payload()
    DOCS_DATA.parent.mkdir(parents=True, exist_ok=True)
    DOCS_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    DOCS_LIVE.write_text(build_live_html(payload), encoding="utf-8")
    print(f"Wrote {DOCS_DATA}")
    print(f"Wrote {DOCS_LIVE}")


if __name__ == "__main__":
    main()
