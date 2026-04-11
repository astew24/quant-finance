const THEME = {
  ink: "#0f1a24",
  soft: "#243240",
  teal: "#0c6a66",
  copper: "#b5662a",
  rose: "#9b3f34",
  green: "#176548",
  gold: "#8f6426",
};

const state = {
  data: null,
  selectedCrypto: "BTC-USD",
  options: null,
};

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

const monthFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  year: "2-digit",
});

function formatDate(value) {
  return dateFormatter.format(new Date(`${value}T00:00:00`));
}

function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPercent(value, digits = 1) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function formatSignedNumber(value, digits = 2) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${Number(value).toFixed(digits)}`;
}

function formatCurrency(value, digits = 2) {
  return `$${Number(value).toFixed(digits)}`;
}

function metricCard(label, value, detail) {
  return `
    <div class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-detail">${detail}</div>
    </div>
  `;
}

function niceTicks(min, max, count = 4) {
  if (min === max) {
    return [min];
  }

  const rawStep = Math.abs(max - min) / Math.max(count - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;

  let niceStep = magnitude;
  if (residual >= 5) {
    niceStep = 5 * magnitude;
  } else if (residual >= 2) {
    niceStep = 2 * magnitude;
  }

  const tickMin = Math.floor(min / niceStep) * niceStep;
  const tickMax = Math.ceil(max / niceStep) * niceStep;
  const ticks = [];

  for (let tick = tickMin; tick <= tickMax + niceStep * 0.5; tick += niceStep) {
    ticks.push(Number(tick.toFixed(8)));
  }

  return ticks;
}

function normalizeSeries(series, xType) {
  return series.map((item) => ({
    ...item,
    points: item.points.map((point) => ({
      rawX: point.x,
      x:
        xType === "date"
          ? new Date(`${point.x}T00:00:00`).getTime()
          : Number(point.x),
      y: point.y == null || Number.isNaN(point.y) ? null : Number(point.y),
    })),
  }));
}

function buildLinePath(points, xScale, yScale) {
  let path = "";
  let open = false;
  for (const point of points) {
    if (point.y == null) {
      open = false;
      continue;
    }
    const x = xScale(point.x);
    const y = yScale(point.y);
    path += `${open ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    open = true;
  }
  return path;
}

function buildAreaPath(points, xScale, yScale, baseline) {
  const filtered = points.filter((point) => point.y != null);
  if (!filtered.length) {
    return "";
  }

  const top = filtered
    .map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`)
    .join("");
  const bottom = filtered
    .slice()
    .reverse()
    .map((point) => `L${xScale(point.x).toFixed(2)},${baseline.toFixed(2)}`)
    .join("");

  return `${top}${bottom}Z`;
}

function renderLegend(series) {
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = series
    .map(
      (item) => `
        <span class="legend-chip">
          <span class="legend-swatch" style="background:${item.color}"></span>
          <span>${item.name}</span>
        </span>
      `
    )
    .join("");
  return legend;
}

function renderLineChart(target, config) {
  const width = Math.max(target.clientWidth || 0, 320);
  const height = config.height ?? 300;
  const margin = { top: 18, right: 16, bottom: 36, left: 52 };
  const series = normalizeSeries(config.series, config.xType);
  const allPoints = series.flatMap((item) => item.points.filter((point) => point.y != null));

  if (!allPoints.length) {
    target.innerHTML = `<p class="card-note">No data available for this chart.</p>`;
    return;
  }

  let minX = Math.min(...allPoints.map((point) => point.x));
  let maxX = Math.max(...allPoints.map((point) => point.x));
  let minY = Math.min(...allPoints.map((point) => point.y));
  let maxY = Math.max(...allPoints.map((point) => point.y));

  if (config.baselineZero) {
    minY = Math.min(minY, 0);
    maxY = Math.max(maxY, 0);
  }

  const yPad = Math.max((maxY - minY) * 0.12, (config.baselineZero ? 0 : 1) * 0.001);
  minY -= yPad;
  maxY += yPad;

  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }

  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const xScale = (value) => margin.left + ((value - minX) / (maxX - minX)) * plotWidth;
  const yScale = (value) => margin.top + plotHeight - ((value - minY) / (maxY - minY)) * plotHeight;

  const yTicks = niceTicks(minY, maxY, 5);
  const xTicks = [];
  const xTickCount = 4;
  for (let index = 0; index < xTickCount; index += 1) {
    const ratio = xTickCount === 1 ? 0 : index / (xTickCount - 1);
    xTicks.push(minX + (maxX - minX) * ratio);
  }

  const svgParts = [];
  svgParts.push(`<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${config.label ?? "chart"}">`);

  for (const tick of yTicks) {
    const y = yScale(tick);
    svgParts.push(`<line class="chart-grid-line" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>`);
    svgParts.push(
      `<text class="chart-label" x="${margin.left - 10}" y="${y + 4}" text-anchor="end">${config.yFormat ? config.yFormat(tick) : formatNumber(tick, 2)}</text>`
    );
  }

  for (const tick of xTicks) {
    const x = xScale(tick);
    const label =
      config.xType === "date"
        ? monthFormatter.format(new Date(tick))
        : config.xFormat
          ? config.xFormat(tick)
          : formatNumber(tick, 2);
    svgParts.push(`<line class="chart-grid-line" x1="${x}" y1="${margin.top}" x2="${x}" y2="${height - margin.bottom}"></line>`);
    svgParts.push(
      `<text class="chart-label" x="${x}" y="${height - 10}" text-anchor="middle">${label}</text>`
    );
  }

  if (config.horizontalLines) {
    for (const line of config.horizontalLines) {
      const y = yScale(line.value);
      svgParts.push(
        `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${line.color}" stroke-width="1.5" stroke-dasharray="5 5"></line>`
      );
      svgParts.push(
        `<text class="chart-label" x="${width - margin.right}" y="${y - 6}" text-anchor="end">${line.label}</text>`
      );
    }
  }

  if (config.verticalLines) {
    for (const line of config.verticalLines) {
      const xValue = config.xType === "date" ? new Date(`${line.value}T00:00:00`).getTime() : Number(line.value);
      const x = xScale(xValue);
      svgParts.push(
        `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${height - margin.bottom}" stroke="${line.color}" stroke-width="1.5" stroke-dasharray="5 5"></line>`
      );
      svgParts.push(
        `<text class="chart-label" x="${x + 6}" y="${margin.top + 12}" text-anchor="start">${line.label}</text>`
      );
    }
  }

  for (const item of series) {
    if (item.fill) {
      const area = buildAreaPath(item.points, xScale, yScale, margin.top + plotHeight);
      if (area) {
        svgParts.push(`<path d="${area}" fill="${item.fill}" opacity="${item.fillOpacity ?? 1}"></path>`);
      }
    }
  }

  for (const item of series) {
    const path = buildLinePath(item.points, xScale, yScale);
    svgParts.push(
      `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="${item.strokeWidth ?? 3}" stroke-linecap="round" stroke-linejoin="round"${item.dashed ? ` stroke-dasharray="${item.dashed}"` : ""}></path>`
    );
  }

  svgParts.push(
    `<line class="chart-axis-line" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"></line>`
  );
  svgParts.push("</svg>");

  target.innerHTML = svgParts.join("");
  target.appendChild(renderLegend(config.series));
}

function renderBarChart(target, config) {
  const width = Math.max(target.clientWidth || 0, 320);
  const margin = { top: 12, right: 16, bottom: 28, left: 128 };
  const data = config.data;
  const height = Math.max(220, data.length * 52);

  const minX = Math.min(0, ...data.map((item) => item.value));
  const maxX = Math.max(0, ...data.map((item) => item.value));
  const pad = Math.max((maxX - minX) * 0.12, 0.05);
  const domainMin = minX - pad;
  const domainMax = maxX + pad;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const rowHeight = plotHeight / data.length;
  const xScale = (value) => margin.left + ((value - domainMin) / (domainMax - domainMin)) * plotWidth;
  const zeroX = xScale(0);
  const ticks = niceTicks(domainMin, domainMax, 5);

  const svgParts = [];
  svgParts.push(`<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${config.label ?? "bar chart"}">`);

  for (const tick of ticks) {
    const x = xScale(tick);
    svgParts.push(`<line class="chart-grid-line" x1="${x}" y1="${margin.top}" x2="${x}" y2="${height - margin.bottom}"></line>`);
    svgParts.push(
      `<text class="chart-label" x="${x}" y="${height - 8}" text-anchor="middle">${config.xFormat ? config.xFormat(tick) : formatNumber(tick, 2)}</text>`
    );
  }

  data.forEach((item, index) => {
    const y = margin.top + index * rowHeight + rowHeight * 0.2;
    const barHeight = rowHeight * 0.58;
    const x = item.value >= 0 ? zeroX : xScale(item.value);
    const barWidth = Math.abs(xScale(item.value) - zeroX);
    svgParts.push(
      `<text class="chart-label" x="${margin.left - 10}" y="${y + barHeight / 2 + 4}" text-anchor="end">${item.label}</text>`
    );
    svgParts.push(
      `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="8" fill="${item.color}"></rect>`
    );
  });

  svgParts.push(`<line class="chart-axis-line" x1="${zeroX}" y1="${margin.top}" x2="${zeroX}" y2="${height - margin.bottom}"></line>`);
  svgParts.push("</svg>");
  target.innerHTML = svgParts.join("");
}

function buildSignalCard({ eyebrow, title, body, footer }) {
  return `
    <p class="eyebrow">${eyebrow}</p>
    <strong>${title}</strong>
    <p>${body}</p>
    <small>${footer}</small>
  `;
}

function renderHero(data) {
  document.getElementById("repo-link").href = data.repo_url;
  document.getElementById("live-link").href = data.live_url;

  document.getElementById("hero-metrics").innerHTML = data.hero_metrics
    .map(
      (item) => `
        <div class="hero-metric">
          <div class="hero-metric-label">${item.label}</div>
          <div class="hero-metric-value">${item.value}</div>
          <div class="hero-metric-detail">${item.detail}</div>
        </div>
      `
    )
    .join("");

  const btc = data.crypto.assets.find((asset) => asset.symbol === "BTC-USD");
  const factor = data.factor;
  const options = data.options.sample_output;

  document.getElementById("signal-crypto").innerHTML = buildSignalCard({
    eyebrow: "Crypto",
    title: "BTC and ETH sample window stays explicit",
    body: `Volatility study covers ${formatDate(btc.date_range.start)} to ${formatDate(btc.date_range.end)} with walk-forward validation and overlay diagnostics.`,
    footer: `BTC RMSE ${formatNumber(btc.metrics.walk_forward_rmse, 3)} vs random walk ${formatNumber(btc.metrics.random_walk_rmse, 3)}`,
  });

  document.getElementById("signal-factor").innerHTML = buildSignalCard({
    eyebrow: "Factor",
    title: "Backtest, attribution, and screen live together",
    body: `Research window runs ${formatDate(factor.metadata.start_date)} to ${formatDate(factor.metadata.end_date)} across ${factor.metadata.backtest_universe_size} backtest names and ${factor.metadata.screening_universe_size} screened names.`,
    footer: `Total return ${formatPercent(factor.metrics.total_return, 1)} / mean IC ${formatNumber(factor.metrics.mean_information_coefficient, 3)}`,
  });

  document.getElementById("signal-options").innerHTML = buildSignalCard({
    eyebrow: "Options",
    title: "The live link keeps a real interactive demo",
    body: "The browser pricing module recalculates payoff, volatility sensitivity, and comparative methods without calling a backend.",
    footer: `Sample call price ${formatNumber(options.black_scholes_price, 4)} / American tree ${formatNumber(options.american_tree_price, 4)}`,
  });

  document.getElementById("review-repo").href = data.repo_url;
  document.getElementById("review-crypto").href = data.project_links.crypto;
  document.getElementById("review-factor").href = data.project_links.factor;
  document.getElementById("review-options").href = data.project_links.options;
}

function renderCryptoToggles() {
  const container = document.getElementById("crypto-toggle");
  container.innerHTML = state.data.crypto.assets
    .map(
      (asset) => `
        <button class="toggle-button ${asset.symbol === state.selectedCrypto ? "is-active" : ""}" type="button" data-symbol="${asset.symbol}">
          ${asset.label}
        </button>
      `
    )
    .join("");

  container.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedCrypto = button.dataset.symbol;
      renderCrypto();
    });
  });
}

function renderCrypto() {
  renderCryptoToggles();

  const asset = state.data.crypto.assets.find((item) => item.symbol === state.selectedCrypto);
  const lastVolPoint = [...asset.vol_series].reverse().find((item) => item.conditional_vol != null);

  document.getElementById("crypto-dates").textContent = `${formatDate(asset.date_range.start)} to ${formatDate(asset.date_range.end)}`;

  document.getElementById("crypto-metrics").innerHTML = [
    metricCard("Annualized volatility", formatPercent(asset.metrics.annual_volatility, 1), "Sample annualized realized volatility."),
    metricCard("VaR 95%", formatSignedPercent(asset.metrics.var_95, 2), "Empirical daily loss threshold."),
    metricCard(
      "Walk-forward RMSE",
      formatNumber(asset.metrics.walk_forward_rmse, 3),
      `Random walk baseline ${formatNumber(asset.metrics.random_walk_rmse, 3)}`
    ),
    metricCard("Direction accuracy", formatPercent(asset.metrics.direction_accuracy, 1), "Sign accuracy over out-of-sample windows."),
    metricCard("Overlay Sharpe", formatSignedNumber(asset.metrics.strategy_sharpe, 2), "Vol-managed overlay sample performance."),
    metricCard("Avg leverage", `${formatNumber(asset.metrics.strategy_average_leverage, 2)}x`, `Max drawdown ${formatPercent(asset.metrics.strategy_max_drawdown, 1)}`),
  ].join("");

  renderLineChart(document.getElementById("crypto-price-chart"), {
    xType: "date",
    label: `${asset.label} normalized price chart`,
    yFormat: (value) => value.toFixed(0),
    series: [
      {
        name: `${asset.label} price index`,
        color: THEME.teal,
        fill: "rgba(12, 106, 102, 0.12)",
        points: asset.price_series.map((point) => ({ x: point.date, y: point.price_index })),
      },
    ],
  });

  const forecastPoints = lastVolPoint
    ? [{ x: lastVolPoint.date, y: lastVolPoint.conditional_vol }, ...asset.forecast_series.map((point) => ({ x: point.date, y: point.forecast_vol }))]
    : asset.forecast_series.map((point) => ({ x: point.date, y: point.forecast_vol }));

  renderLineChart(document.getElementById("crypto-vol-chart"), {
    xType: "date",
    label: `${asset.label} volatility chart`,
    yFormat: (value) => formatPercent(value, 1),
    series: [
      {
        name: "Realized vol",
        color: THEME.copper,
        points: asset.vol_series.map((point) => ({ x: point.date, y: point.realized_vol })),
      },
      {
        name: "Conditional vol",
        color: THEME.gold,
        points: asset.vol_series.map((point) => ({ x: point.date, y: point.conditional_vol })),
      },
      {
        name: "Forward forecast",
        color: THEME.ink,
        dashed: "7 6",
        points: forecastPoints,
      },
    ],
  });

  renderLineChart(document.getElementById("crypto-overlay-chart"), {
    xType: "date",
    label: `${asset.label} overlay chart`,
    yFormat: (value) => formatPercent(value, 0),
    baselineZero: true,
    series: [
      {
        name: "Vol-managed overlay",
        color: THEME.teal,
        points: asset.overlay_series.map((point) => ({ x: point.date, y: point.strategy_curve })),
      },
      {
        name: "Buy and hold",
        color: THEME.soft,
        dashed: "5 5",
        points: asset.overlay_series.map((point) => ({ x: point.date, y: point.benchmark_curve })),
      },
    ],
  });
}

function renderFactor() {
  const factor = state.data.factor;
  document.getElementById("factor-dates").textContent = `${formatDate(factor.metadata.start_date)} to ${formatDate(factor.metadata.end_date)} / ${factor.metadata.transaction_cost_bps} bps transaction cost`;

  document.getElementById("factor-metrics").innerHTML = [
    metricCard("Total return", formatPercent(factor.metrics.total_return, 1), "Long-short strategy total return."),
    metricCard("Sharpe ratio", formatNumber(factor.metrics.sharpe_ratio, 2), "Risk-adjusted backtest performance."),
    metricCard("Mean IC", formatNumber(factor.metrics.mean_information_coefficient, 3), "Average cross-sectional information coefficient."),
    metricCard("Holdout accuracy", formatPercent(factor.metrics.holdout_accuracy, 1), "Classifier holdout accuracy."),
    metricCard("ROC AUC", formatNumber(factor.metrics.holdout_roc_auc, 3), "Holdout ranking quality."),
    metricCard("Screen size", `${factor.metadata.screening_universe_size} names`, `${factor.metadata.backtest_universe_size} names in backtest universe`),
  ].join("");

  renderLineChart(document.getElementById("factor-performance-chart"), {
    xType: "date",
    label: "Factor strategy curve",
    yFormat: (value) => formatPercent(value, 0),
    baselineZero: true,
    series: [
      {
        name: "Long-short strategy",
        color: THEME.teal,
        points: factor.performance_series.map((point) => ({ x: point.date, y: point.strategy_curve })),
      },
      {
        name: "Equal-weight benchmark",
        color: THEME.soft,
        dashed: "5 5",
        points: factor.performance_series.map((point) => ({ x: point.date, y: point.benchmark_curve })),
      },
    ],
  });

  renderBarChart(document.getElementById("factor-exposure-chart"), {
    label: "Factor exposure chart",
    xFormat: (value) => formatNumber(value, 2),
    data: factor.factor_exposures.map((item) => ({
      label: item.factor,
      value: item.beta,
      color: item.beta >= 0 ? THEME.green : THEME.rose,
    })),
  });

  document.getElementById("factor-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Ticker</th>
          <th>Sector</th>
          <th>Screen</th>
          <th>ML Prob.</th>
          <th>Price</th>
          <th>Target</th>
        </tr>
      </thead>
      <tbody>
        ${factor.top_screen
          .map(
            (row) => `
              <tr>
                <td>${row.screen_rank}</td>
                <td><strong>${row.ticker}</strong></td>
                <td>${row.sector}</td>
                <td>${formatNumber(row.screen_score, 3)}</td>
                <td>${row.probability.toFixed(1)}%</td>
                <td>${formatCurrency(row.price)}</td>
                <td>${formatCurrency(row.target)}</td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;

  document.getElementById("factor-ideas").innerHTML = factor.ideas
    .map(
      (idea) => `
        <article class="idea-card">
          <p class="section-index">${idea.ticker}</p>
          <h3>${idea.ticker} thesis snapshot</h3>
          <p>${idea.thesis}</p>
          <p class="idea-risk"><strong>Risk:</strong> ${idea.risk}</p>
        </article>
      `
    )
    .join("");
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX));
  return sign * y;
}

function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function normPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function blackScholes(spot, strike, rate, volatility, maturity, optionType) {
  const sqrtT = Math.sqrt(maturity);
  const d1 = (Math.log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity) / (volatility * sqrtT);
  const d2 = d1 - volatility * sqrtT;
  const discount = Math.exp(-rate * maturity);

  let price;
  let delta;
  let theta;
  let rho;

  if (optionType === "call") {
    price = spot * normCdf(d1) - strike * discount * normCdf(d2);
    delta = normCdf(d1);
    theta =
      (-spot * normPdf(d1) * volatility) / (2 * sqrtT) -
      rate * strike * discount * normCdf(d2);
    rho = strike * maturity * discount * normCdf(d2);
  } else {
    price = strike * discount * normCdf(-d2) - spot * normCdf(-d1);
    delta = normCdf(d1) - 1;
    theta =
      (-spot * normPdf(d1) * volatility) / (2 * sqrtT) +
      rate * strike * discount * normCdf(-d2);
    rho = -strike * maturity * discount * normCdf(-d2);
  }

  return {
    price,
    delta,
    gamma: normPdf(d1) / (spot * volatility * sqrtT),
    vega: spot * normPdf(d1) * sqrtT,
    theta,
    rho,
  };
}

function americanOptionBinomial(spot, strike, rate, volatility, maturity, optionType, steps = 200) {
  const dt = maturity / steps;
  const up = Math.exp(volatility * Math.sqrt(dt));
  const down = 1 / up;
  const growth = Math.exp(rate * dt);
  const prob = Math.min(1, Math.max(0, (growth - down) / (up - down)));
  const discount = Math.exp(-rate * dt);
  const values = [];

  for (let i = 0; i <= steps; i += 1) {
    const underlying = spot * (up ** (steps - i)) * (down ** i);
    values[i] = optionType === "call" ? Math.max(underlying - strike, 0) : Math.max(strike - underlying, 0);
  }

  for (let step = steps - 1; step >= 0; step -= 1) {
    for (let node = 0; node <= step; node += 1) {
      const continuation = discount * (prob * values[node] + (1 - prob) * values[node + 1]);
      const underlying = spot * (up ** (step - node)) * (down ** node);
      const exercise = optionType === "call" ? Math.max(underlying - strike, 0) : Math.max(strike - underlying, 0);
      values[node] = Math.max(continuation, exercise);
    }
  }

  return values[0];
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let value = Math.imul(t ^ (t >>> 15), 1 | t);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function monteCarloPrice(spot, strike, rate, volatility, maturity, optionType, paths = 6000, seed = 42) {
  const random = mulberry32(seed);
  let spare = null;
  const normal = () => {
    if (spare != null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    spare = mag * Math.sin(2.0 * Math.PI * v);
    return mag * Math.cos(2.0 * Math.PI * v);
  };

  const discount = Math.exp(-rate * maturity);
  let sum = 0;
  let sumSq = 0;

  for (let index = 0; index < paths; index += 1) {
    const z = normal();
    const terminal = spot * Math.exp((rate - 0.5 * volatility * volatility) * maturity + volatility * Math.sqrt(maturity) * z);
    const payoff = optionType === "call" ? Math.max(terminal - strike, 0) : Math.max(strike - terminal, 0);
    const discounted = discount * payoff;
    sum += discounted;
    sumSq += discounted * discounted;
  }

  const price = sum / paths;
  const variance = Math.max((sumSq - (sum * sum) / paths) / Math.max(paths - 1, 1), 0);
  const stdError = Math.sqrt(variance / paths);

  return {
    price,
    ci: 1.96 * stdError,
  };
}

function renderOptionsControls() {
  const config = [
    { key: "spot", label: "Spot price", min: 50, max: 150, step: 1, format: (value) => formatCurrency(value, 0) },
    { key: "strike", label: "Strike", min: 50, max: 150, step: 1, format: (value) => formatCurrency(value, 0) },
    { key: "maturity", label: "Maturity", min: 0.1, max: 2, step: 0.05, format: (value) => `${value.toFixed(2)}y` },
    { key: "rate", label: "Risk-free rate", min: 0, max: 0.1, step: 0.0025, format: (value) => formatPercent(value, 1) },
    { key: "volatility", label: "Volatility", min: 0.05, max: 0.8, step: 0.01, format: (value) => formatPercent(value, 0) },
  ];

  const container = document.getElementById("options-controls");
  container.innerHTML = config
    .map(
      (item) => `
        <label class="slider-row" for="control-${item.key}">
          <span class="slider-meta">
            <span>${item.label}</span>
            <span id="control-value-${item.key}">${item.format(state.options[item.key])}</span>
          </span>
          <input
            id="control-${item.key}"
            type="range"
            min="${item.min}"
            max="${item.max}"
            step="${item.step}"
            value="${state.options[item.key]}"
            data-key="${item.key}"
          >
        </label>
      `
    )
    .join("");

  container.querySelectorAll("input").forEach((input) => {
    const item = config.find((entry) => entry.key === input.dataset.key);
    input.addEventListener("input", () => {
      state.options[input.dataset.key] = Number(input.value);
      document.getElementById(`control-value-${input.dataset.key}`).textContent = item.format(state.options[input.dataset.key]);
      renderOptions();
    });
  });

  document.querySelectorAll("[data-option-type]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.optionType === state.options.optionType);
    button.addEventListener("click", () => {
      state.options.optionType = button.dataset.optionType;
      document.querySelectorAll("[data-option-type]").forEach((entry) => {
        entry.classList.toggle("is-active", entry.dataset.optionType === state.options.optionType);
      });
      renderOptions();
    });
  });
}

function renderOptions() {
  const { optionType, spot, strike, rate, volatility, maturity } = state.options;
  const bs = blackScholes(spot, strike, rate, volatility, maturity, optionType);
  const american = americanOptionBinomial(spot, strike, rate, volatility, maturity, optionType, 180);
  const mc = monteCarloPrice(spot, strike, rate, volatility, maturity, optionType, 5000, 42);

  const priceGrid = [];
  for (let x = spot * 0.45; x <= spot * 1.55; x += (spot * 1.1) / 44) {
    const intrinsic = optionType === "call" ? Math.max(x - strike, 0) : Math.max(strike - x, 0);
    priceGrid.push({ x, intrinsic, pnl: intrinsic - bs.price });
  }

  const breakeven = optionType === "call" ? strike + bs.price : strike - bs.price;
  renderLineChart(document.getElementById("options-payoff-chart"), {
    xType: "number",
    label: "Options payoff chart",
    yFormat: (value) => formatCurrency(value, 0),
    xFormat: (value) => formatCurrency(value, 0),
    baselineZero: true,
    verticalLines: [{ value: breakeven, label: `Breakeven ${formatCurrency(breakeven, 1)}`, color: THEME.rose }],
    series: [
      {
        name: "Intrinsic payoff",
        color: THEME.teal,
        points: priceGrid.map((point) => ({ x: point.x, y: point.intrinsic })),
      },
      {
        name: "Net P/L",
        color: THEME.copper,
        dashed: "7 6",
        points: priceGrid.map((point) => ({ x: point.x, y: point.pnl })),
      },
    ],
  });

  const volGrid = [];
  for (let sigma = 0.05; sigma <= 0.8; sigma += 0.015) {
    volGrid.push({
      x: sigma * 100,
      y: blackScholes(spot, strike, rate, sigma, maturity, optionType).price,
    });
  }

  renderLineChart(document.getElementById("options-vol-chart"), {
    xType: "number",
    label: "Option value vs volatility chart",
    yFormat: (value) => formatCurrency(value, 0),
    xFormat: (value) => `${value.toFixed(0)}%`,
    verticalLines: [{ value: volatility * 100, label: `Selected ${formatPercent(volatility, 0)}`, color: THEME.teal }],
    series: [
      {
        name: "Black-Scholes value",
        color: THEME.ink,
        points: volGrid,
      },
    ],
  });

  document.getElementById("options-metrics").innerHTML = [
    metricCard("Black-Scholes price", formatCurrency(bs.price, 4), "Closed-form European price."),
    metricCard("American tree", formatCurrency(american, 4), "CRR binomial tree approximation."),
    metricCard("Monte Carlo", formatCurrency(mc.price, 4), `95% CI +/- ${mc.ci.toFixed(4)}`),
    metricCard("Delta", formatSignedNumber(bs.delta, 4), "First derivative wrt spot."),
    metricCard("Gamma", formatNumber(bs.gamma, 4), "Curvature of delta."),
    metricCard("Vega", formatNumber(bs.vega, 4), "Sensitivity to implied volatility."),
    metricCard("Theta", formatSignedNumber(bs.theta, 4), "Time-decay estimate."),
    metricCard("Rho", formatSignedNumber(bs.rho, 4), "Rate sensitivity."),
  ].join("");
}

function debounce(fn, wait = 150) {
  let timeout = null;
  return (...args) => {
    window.clearTimeout(timeout);
    timeout = window.setTimeout(() => fn(...args), wait);
  };
}

function rerenderAll() {
  renderCrypto();
  renderFactor();
  renderOptions();
}

async function init() {
  const inlineData = document.getElementById("portfolio-data");
  const data = inlineData
    ? JSON.parse(inlineData.textContent)
    : await fetch("./assets/data/portfolio.json").then((response) => response.json());
  state.data = data;
  state.options = { ...data.options.default_scenario };

  renderHero(data);
  renderCrypto();
  renderFactor();
  renderOptionsControls();
  renderOptions();

  window.addEventListener("resize", debounce(rerenderAll));
}

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div class="noscript-banner">The portfolio data could not be loaded.</div>`
  );
});
