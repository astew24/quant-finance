"""Plotting utilities for pipeline results."""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_results(results: Dict, output_dir: str = "output") -> None:
    """Save per-symbol analysis charts and a forecast comparison."""
    os.makedirs(output_dir, exist_ok=True)

    for sym, r in results.items():
        safe = sym.replace("/", "_").replace("-", "_")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        fig.suptitle(f"{sym} -- Volatility Analysis", fontsize=14, fontweight="bold")

        # price
        axes[0].plot(r.prices, linewidth=0.8)
        axes[0].set_title("Price (USD)")
        axes[0].set_ylabel("Price")
        axes[0].grid(alpha=0.3)

        # returns
        axes[1].bar(r.returns.index, r.returns.values, width=1,
                     color="steelblue", alpha=0.6)
        axes[1].set_title("Daily Log Returns")
        axes[1].set_ylabel("Return")
        axes[1].grid(alpha=0.3)

        # volatility
        rv = r.realised_vol.dropna()
        if not rv.empty:
            axes[2].plot(rv, label="Realised (30d)", linewidth=0.9, alpha=0.8)
        if not r.conditional_vol.empty:
            axes[2].plot(r.conditional_vol, label="GARCH conditional",
                         linewidth=0.9, alpha=0.8)
        axes[2].set_title("Volatility")
        axes[2].set_ylabel("Annualised vol")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        path = os.path.join(output_dir, f"{safe}_analysis.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot -> {path}")

    # multi-symbol forecast comparison
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        for sym, r in results.items():
            if not r.forecast.empty:
                ax.plot(r.forecast, marker="o", label=sym)
        ax.set_title("Volatility Forecast Comparison")
        ax.set_ylabel("Forecasted daily vol")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "forecast_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot -> {path}")
