import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "data")

factor_tickers = [
    "SPY",
    "ACWI",
    "TLT",
    "HYG",
    "DBC",
    "EEM",
    "UUP",
    "TIP",
    "SVXY",
    "SHY",
    "USMV",
    "MTUM",
    "QUAL",
    "IVE",
    "IWM",
    "GLD",
    "USO",
    "VIXY",
    "^TNX",
    "^IRX",
]

rename_map = {
    "TLT": "Interest Rates",
    "HYG": "Credit",
    "DBC": "Commodities",
    "EEM": "Emerging Markets",
    "UUP": "FX",
    "TIP": "Real Yields",
    "SVXY": "Equity Short Vol",
    "USMV": "Low Risk",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "IVE": "Value",
    "IWM": "Small Cap",
    "ACWI": "Global Equity",
    "GLD": "Gold",
    "USO": "Oil",
    "VIXY": "Volatility",
}

factor_cols = [
    "Global Equity",
    "Interest Rates",
    "Credit",
    "Commodities",
    "Emerging Markets",
    "FX",
    "Real Yields",
    "Local Inflation",
    "Equity Short Vol",
    "FX Carry",
    "Trend",
    "Low Risk",
    "Momentum",
    "Quality",
    "Value",
    "Small Cap",
    "Gold",
    "Oil",
    "Volatility",
    "FI Carry",
]

PLOTLY_THEME = "plotly_dark"


# =========================
# DATA HELPERS
# =========================

def completed_month_end() -> pd.Timestamp:
    """
    Return the current calendar month-end timestamp.

    Filtering with index < this value keeps only completed months.
    Example: on August 3, July 31 remains and August 31 is excluded.
    """
    return pd.Timestamp.today().to_period("M").to_timestamp("M")


def load_prices_from_csv(ticker: str) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    except (ValueError, KeyError, pd.errors.ParserError):
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    return df


def download_prices(tickers) -> pd.DataFrame:
    dfs = []

    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        df = load_prices_from_csv(ticker)

        if df.empty:
            continue

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        if price_col not in df.columns:
            continue

        price_series = pd.to_numeric(df[price_col], errors="coerce")
        dfs.append(price_series.rename(ticker).to_frame())

    if not dfs:
        return pd.DataFrame()

    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


@st.cache_data
def prepare_factors() -> pd.DataFrame:
    price_df = download_prices(factor_tickers)

    if price_df.empty:
        return pd.DataFrame()

    # Month-end observations, labeled by the calendar month-end date.
    price_df = price_df.resample("ME").last()
    price_df = price_df[price_df.index < completed_month_end()]

    raw_rets = price_df.pct_change().dropna(how="all")
    factors = raw_rets.rename(columns=rename_map).copy()

    if "EEM" in raw_rets.columns and "UUP" in raw_rets.columns:
        factors["FX Carry"] = raw_rets["EEM"] - raw_rets["UUP"]
    else:
        factors["FX Carry"] = np.nan

    if "TIP" in raw_rets.columns and "TLT" in raw_rets.columns:
        tip, tlt = raw_rets["TIP"].align(raw_rets["TLT"], join="inner")
        factors.loc[tip.index, "Local Inflation"] = tip - tlt
    else:
        factors["Local Inflation"] = np.nan

    if "SPY" in price_df.columns:
        factors["Trend"] = price_df["SPY"].pct_change(12)
    else:
        factors["Trend"] = np.nan

    # FI Carry proxy: monthly change in the 10Y-minus-3M yield spread.
    tnx = load_prices_from_csv("^TNX")
    irx = load_prices_from_csv("^IRX")

    if not tnx.empty and not irx.empty:
        tnx_col = "Adj Close" if "Adj Close" in tnx.columns else "Close"
        irx_col = "Adj Close" if "Adj Close" in irx.columns else "Close"

        tnx_yield = (
            pd.to_numeric(tnx[tnx_col], errors="coerce") / 100.0
        ).resample("ME").last()

        irx_yield = (
            pd.to_numeric(irx[irx_col], errors="coerce") / 100.0
        ).resample("ME").last()

        fi_carry = (tnx_yield - irx_yield).diff()
        factors["FI Carry"] = fi_carry.reindex(factors.index)
    else:
        factors["FI Carry"] = np.nan

    keep = [column for column in factor_cols if column in factors.columns]
    return factors[keep]


def get_rf(index: pd.Index) -> pd.Series:
    """
    Approximate monthly risk-free return using the 3-month Treasury yield.
    """
    df = load_prices_from_csv("^IRX")

    if df.empty:
        return pd.Series(0.0, index=index, name="RF")

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    rf = (
        pd.to_numeric(df[price_col], errors="coerce") / 100.0 / 12.0
    ).resample("ME").last()

    return (
        rf.reindex(index, method="ffill")
        .fillna(0.0)
        .rename("RF")
    )


@st.cache_data
def load_and_merge_all_data(fund_tickers: tuple):
    factors = prepare_factors()

    if factors.empty:
        return None

    fund_prices = download_prices(list(fund_tickers))

    if fund_prices.empty:
        return None

    fund_prices = fund_prices.resample("ME").last()
    fund_prices = fund_prices[fund_prices.index < completed_month_end()]

    fund_rets = fund_prices.pct_change().dropna(how="all")

    if fund_rets.empty:
        return None

    # Use only months shared by the fund and all retained factors.
    df = fund_rets.join(factors, how="inner").dropna()

    if df.empty:
        return None

    rf = get_rf(df.index)
    df["RF"] = rf

    for fund in fund_rets.columns:
        if fund in df.columns:
            df[f"{fund}_Excess"] = df[fund] - df["RF"]

    return df


# =========================
# FACTOR REGRESSIONS
# =========================

def compute_static(df: pd.DataFrame, fund: str):
    cols = [column for column in factor_cols if column in df.columns]

    if not cols or f"{fund}_Excess" not in df.columns:
        return None, None, None, None

    model_df = df[[f"{fund}_Excess"] + cols].dropna()

    if model_df.empty:
        return None, None, None, None

    X = sm.add_constant(model_df[cols])
    y = model_df[f"{fund}_Excess"]

    model = sm.OLS(y, X).fit()

    betas = model.params.drop("const", errors="ignore")
    t_values = model.tvalues.drop("const", errors="ignore")
    alpha_monthly = float(model.params.get("const", np.nan))
    r_squared = float(model.rsquared)

    return (
        betas.round(3),
        t_values.round(2),
        r_squared,
        alpha_monthly,
    )


def compute_rolling(
    df: pd.DataFrame,
    fund: str,
    window: int = 36,
) -> pd.DataFrame:
    cols = [column for column in factor_cols if column in df.columns]
    required = [f"{fund}_Excess"] + cols
    model_df = df[required].dropna()

    if len(model_df) < window:
        return pd.DataFrame()

    y = model_df[f"{fund}_Excess"].to_numpy()
    X = model_df[cols].to_numpy()
    X_full = np.hstack([np.ones((len(X), 1)), X])

    rolling_betas = []

    for i in range(window - 1, len(X_full)):
        X_window = X_full[i - window + 1 : i + 1]
        y_window = y[i - window + 1 : i + 1]
        coefficients, *_ = np.linalg.lstsq(X_window, y_window, rcond=None)
        rolling_betas.append(coefficients[1:])

    index = model_df.index[window - 1 :]
    return pd.DataFrame(rolling_betas, index=index, columns=cols)


# =========================
# PERIOD RETURN ATTRIBUTION
# =========================

def compute_return_attribution(
    df: pd.DataFrame,
    fund: str,
    betas: pd.Series,
    months: int,
):
    """
    Exposure-based attribution over a selected historical period.

    Each monthly factor contribution is:
        beta × realized monthly factor return

    Monthly contributions are then summed over the selected period.
    Actual fund return is compounded. The residual therefore includes
    unexplained return, regression error, alpha, and compounding effects.
    """
    cols = [column for column in betas.index if column in df.columns]

    if len(df) < months or not cols or fund not in df.columns:
        return None, None, None

    period_df = df.iloc[-months:].copy()

    monthly_factor_contributions = period_df[cols].mul(
        betas[cols],
        axis=1,
    )

    factor_contributions = monthly_factor_contributions.sum()
    factor_contributions.name = "Contribution"

    actual_return = (1.0 + period_df[fund]).prod() - 1.0
    explained_return = factor_contributions.sum()
    residual = actual_return - explained_return

    gross_contribution = factor_contributions.abs().sum()

    if gross_contribution > 0:
        pct_of_gross = (
            factor_contributions / gross_contribution * 100.0
        ).rename("% of Gross Contribution")
    else:
        pct_of_gross = pd.Series(
            np.nan,
            index=factor_contributions.index,
            name="% of Gross Contribution",
        )

    result = pd.concat(
        [
            betas[cols].rename("Beta"),
            factor_contributions,
            pct_of_gross,
        ],
        axis=1,
    )

    result = result.reindex(
        result["Contribution"].abs().sort_values(ascending=False).index
    )

    monthly_factor_contributions["Explained"] = (
        monthly_factor_contributions.sum(axis=1)
    )
    monthly_factor_contributions["Actual Fund Return"] = period_df[fund]
    monthly_factor_contributions["Monthly Residual"] = (
        period_df[fund] - monthly_factor_contributions["Explained"]
    )

    summary = {
        "start_date": period_df.index.min(),
        "end_date": period_df.index.max(),
        "actual_return": actual_return,
        "explained_return": explained_return,
        "residual": residual,
    }

    return result, summary, monthly_factor_contributions


# =========================
# CHART HELPERS
# =========================

def plot_rolling_heatmap(rolling: pd.DataFrame):
    if rolling.empty:
        return None

    data = rolling.clip(-3, 3)
    order = data.abs().mean().sort_values(ascending=False).index.tolist()
    data = data[order]

    x_labels = [date.strftime("%b %Y") for date in data.index]

    fig = go.Figure(
        go.Heatmap(
            z=data.T.values,
            x=x_labels,
            y=order,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.35, "#fca5a5"],
                [0.5, "#1e293b"],
                [0.65, "#86efac"],
                [1.0, "#15803d"],
            ],
            zmid=0,
            zmin=-2,
            zmax=2,
            colorbar=dict(
                title="Beta",
                thickness=12,
                len=0.8,
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["≤-2", "-1", "0", "1", "≥2"],
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: %{x}<br>"
                "Beta: %{z:.3f}"
                "<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )

    fig.update_layout(
        template=PLOTLY_THEME,
        title="Rolling betas — heatmap",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickangle=-45, nticks=12, title=""),
        yaxis=dict(title="", autorange="reversed"),
        height=max(350, len(order) * 28 + 80),
    )

    return fig


def plot_rolling_filtered(
    rolling: pd.DataFrame,
    selected_factors: list,
):
    if rolling.empty or not selected_factors:
        return None

    data = rolling[selected_factors].clip(-3, 3)

    melted = (
        data.reset_index()
        .rename(columns={data.index.name or "index": "Date"})
        .melt(
            id_vars="Date",
            var_name="Factor",
            value_name="Beta",
        )
    )

    fig = px.line(
        melted,
        x="Date",
        y="Beta",
        color="Factor",
        title=f"Rolling betas — {len(selected_factors)} selected factors",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Month: %{x|%b %Y}<br>"
            "Beta: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.2)",
    )

    fig.update_layout(
        template=PLOTLY_THEME,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")
    return fig


def plot_attribution_bar(
    attr: pd.DataFrame,
    fund: str,
    period_label: str,
):
    colors = [
        "#26a69a" if value >= 0 else "#ef5350"
        for value in attr["Contribution"]
    ]

    fig = go.Figure(
        go.Bar(
            x=attr.index,
            y=attr["Contribution"],
            marker_color=colors,
            text=[
                f"{value:.2%}"
                for value in attr["Contribution"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Contribution: %{y:.3%}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{fund} — factor contributions over {period_label}",
        template=PLOTLY_THEME,
        yaxis_tickformat=".1%",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    return fig


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Factor Attribution Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url(
            'https://fonts.googleapis.com/css2?'
            'family=IBM+Plex+Mono:wght@400;600&'
            'family=IBM+Plex+Sans:wght@300;400;600&display=swap'
        );

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: -0.5px;
        }

        div[data-testid="metric-container"] {
            background: #1a1a2e;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:4px;">
        FACTOR ATTRIBUTION DASHBOARD
    </h1>
    <hr style="border-color:#333; margin:16px 0;">
    """,
    unsafe_allow_html=True,
)

tab_factor, tab_attribution = st.tabs(
    ["Factor Analysis", "Return Attribution"]
)


# =========================
# TAB 1: FACTOR ANALYSIS
# =========================

with tab_factor:
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        fund_ticker = (
            st.text_input(
                "Fund ticker",
                value="",
                placeholder="SPY, AGG, EFA …",
            )
            .strip()
            .upper()
        )

    with c2:
        window = st.slider(
            "Rolling window (months)",
            min_value=12,
            max_value=60,
            value=36,
            step=6,
        )

    with c3:
        top_n = st.slider(
            "Top N variable factors",
            min_value=2,
            max_value=10,
            value=5,
        )

    run = st.button("Run Analysis", type="primary")

    if run and fund_ticker:
        with st.spinner("Loading data and running regressions…"):
            df = load_and_merge_all_data((fund_ticker,))

        if df is None or df.empty:
            st.error(
                f"No usable data for {fund_ticker}. "
                "Check that its CSV exists in the data folder."
            )
        else:
            st.success(
                f"Data loaded for **{fund_ticker}** · "
                f"{df.index.min():%b %Y} → {df.index.max():%b %Y} "
                f"({len(df)} months)"
            )

            betas, t_values, r_squared, alpha_monthly = compute_static(
                df,
                fund_ticker,
            )

            if betas is None:
                st.error("No overlapping factors found for regression.")
            else:
                st.subheader(
                    "Static factor exposures — full sample "
                    f"(R² = {r_squared:.2f})"
                )

                static_table = pd.DataFrame(
                    {
                        "Beta": betas,
                        "t-stat": t_values,
                    }
                )

                static_table = static_table.sort_values(
                    "Beta",
                    key=np.abs,
                    ascending=False,
                )

                def color_beta(value):
                    if value > 0.1:
                        return "color:#52b788"
                    if value < -0.1:
                        return "color:#e63946"
                    return "color:#aaa"

                st.dataframe(
                    static_table.style.format(
                        {
                            "Beta": "{:,.3f}",
                            "t-stat": "{:,.2f}",
                        }
                    ).map(
                        color_beta,
                        subset=["Beta"],
                    ),
                    use_container_width=True,
                )

                st.session_state["betas"] = betas
                st.session_state["fund_ticker"] = fund_ticker
                st.session_state["df"] = df
                st.session_state["alpha_monthly"] = alpha_monthly

            rolling = compute_rolling(
                df,
                fund_ticker,
                window=window,
            )

            if rolling.empty:
                st.warning(
                    f"Not enough history for a {window}-month rolling window."
                )
            else:
                st.session_state["rolling"] = rolling

                st.subheader(
                    f"{window}-month rolling betas — heatmap"
                )

                heatmap = plot_rolling_heatmap(rolling)

                if heatmap:
                    st.plotly_chart(
                        heatmap,
                        use_container_width=True,
                    )

                st.divider()
                st.subheader("Current (last completed month) betas")

                latest_betas = rolling.iloc[-1].sort_values(
                    key=np.abs,
                    ascending=False,
                )

                st.dataframe(
                    latest_betas.to_frame("Beta").style.format(
                        {"Beta": "{:,.3f}"}
                    ),
                    use_container_width=True,
                )

                st.session_state["all_factors"] = (
                    rolling.columns.tolist()
                )

                st.session_state["default_factors"] = (
                    rolling.std()
                    .nlargest(top_n)
                    .index.tolist()
                )

                st.session_state["selected_factors"] = (
                    st.session_state["default_factors"]
                )

    elif run and not fund_ticker:
        st.error("Please enter a fund ticker.")

    if "all_factors" in st.session_state:
        st.divider()
        st.subheader("Factor detail — line view")

        selected = st.multiselect(
            "Select factors to display",
            options=st.session_state["all_factors"],
            key="selected_factors",
            help="Displayed betas are clipped at ±3 to suppress artifacts.",
        )

        if selected and "rolling" in st.session_state:
            line_chart = plot_rolling_filtered(
                st.session_state["rolling"],
                selected,
            )

            if line_chart:
                st.plotly_chart(
                    line_chart,
                    use_container_width=True,
                )


# =========================
# TAB 2: RETURN ATTRIBUTION
# =========================

with tab_attribution:
    st.markdown("### Return Attribution")

    if (
        "betas" not in st.session_state
        or "df" not in st.session_state
        or "fund_ticker" not in st.session_state
    ):
        st.info(
            "Run a Factor Analysis first to unlock attribution."
        )
    else:
        betas = st.session_state["betas"]
        df_ra = st.session_state["df"]
        ticker = st.session_state["fund_ticker"]

        period_label = st.selectbox(
            "Attribution period",
            options=[
                "1 Year",
                "3 Years",
                "5 Years",
                "Since Inception",
            ],
        )

        period_months_map = {
            "1 Year": 12,
            "3 Years": 36,
            "5 Years": 60,
            "Since Inception": len(df_ra),
        }

        period_months = period_months_map[period_label]

        attr, summary, monthly_contributions = (
            compute_return_attribution(
                df_ra,
                ticker,
                betas,
                months=period_months,
            )
        )

        if attr is None:
            st.warning(
                f"Not enough overlapping data for {period_label}."
            )
        else:
            st.caption(
                f"{summary['start_date']:%b %Y} through "
                f"{summary['end_date']:%b %Y}. "
                "Factor contributions use fixed full-sample betas "
                "multiplied by each month's realized factor return."
            )

            metric_1, metric_2, metric_3 = st.columns(3)

            metric_1.metric(
                f"Actual {period_label} Return",
                f"{summary['actual_return']:.2%}",
            )

            metric_2.metric(
                "Summed Factor Contributions",
                f"{summary['explained_return']:.2%}",
            )

            metric_3.metric(
                "Unexplained / Compounding Residual",
                f"{summary['residual']:.2%}",
            )

            st.divider()

            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown(
                    f"**{ticker} — contributions over {period_label}**"
                )

                st.dataframe(
                    attr.style.format(
                        {
                            "Beta": "{:.3f}",
                            "Contribution": "{:.2%}",
                            "% of Gross Contribution": "{:.1f}%",
                        }
                    ),
                    use_container_width=True,
                )

            with col_right:
                st.plotly_chart(
                    plot_attribution_bar(
                        attr,
                        ticker,
                        period_label,
                    ),
                    use_container_width=True,
                )

            with st.expander("Monthly attribution detail"):
                detail_columns = [
                    column
                    for column in monthly_contributions.columns
                    if column not in {
                        "Explained",
                        "Actual Fund Return",
                        "Monthly Residual",
                    }
                ]

                monthly_display = monthly_contributions[
                    detail_columns
                    + [
                        "Explained",
                        "Actual Fund Return",
                        "Monthly Residual",
                    ]
                ].copy()

                monthly_display.index = monthly_display.index.strftime(
                    "%b %Y"
                )

                st.dataframe(
                    monthly_display.style.format("{:.2%}"),
                    use_container_width=True,
                )

            st.caption(
                "The residual is the amount needed to reconcile summed "
                "arithmetic factor contributions with the fund's compounded "
                "period return. It can include alpha, regression error, "
                "omitted exposures, and compounding effects."
            )
