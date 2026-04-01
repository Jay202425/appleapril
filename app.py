import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Stock Forecast (Prophet)",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Apple (AAPL) Stock Price Forecast")
st.markdown("**5-year data · 4-year train / 1-year test · Facebook Prophet · 1-year future forecast**")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    run_button = st.button("🔄 Fetch & Run Forecast", use_container_width=True)
    st.markdown("---")
    st.markdown("**Model tuning**")
    changepoint_prior = st.slider(
        "Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001,
        help="Higher → more flexible trend (risk of overfitting)",
    )
    seasonality_prior = st.slider(
        "Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.01,
        help="Strength of seasonality components",
    )
    yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
    weekly_seasonality = st.checkbox("Weekly seasonality", value=True)

# ── Helper: compute RMSE ──────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ── Main logic (runs on button click or first load) ───────────────────────────
if run_button or "forecast_done" not in st.session_state:

    with st.spinner("Fetching 5 years of AAPL data from Yahoo Finance…"):
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=5 * 365)
        raw = yf.download("AAPL", start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        st.error("Could not fetch data. Check your internet connection.")
        st.stop()

    # ── Preprocessing ────────────────────────────────────────────────────────
    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 may return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Close": "price"})
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # ── Train / Test split ────────────────────────────────────────────────────
    split_date = df.index[-1] - timedelta(days=365)
    train_df   = df[df.index <= split_date].copy()
    test_df    = df[df.index >  split_date].copy()

    # ── Prophet dataframe format ──────────────────────────────────────────────
    def to_prophet(dataframe):
        p = dataframe.reset_index().rename(columns={"Date": "ds", "price": "y"})
        p["ds"] = pd.to_datetime(p["ds"]).dt.tz_localize(None)
        return p[["ds", "y"]]

    train_prophet = to_prophet(train_df)

    # ── Fit model ─────────────────────────────────────────────────────────────
    with st.spinner("Training Prophet model…"):
        model = Prophet(
            changepoint_prior_scale=changepoint_prior,
            seasonality_prior_scale=seasonality_prior,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
        )
        model.fit(train_prophet)

    # ── Predict on test period ────────────────────────────────────────────────
    test_future  = to_prophet(test_df)[["ds"]]
    test_forecast = model.predict(test_future)

    y_true = test_df["price"].values
    y_pred = test_forecast["yhat"].values
    test_rmse = rmse(y_true, y_pred)

    # ── Forecast next 1 year ──────────────────────────────────────────────────
    future_days   = 365
    last_date     = df.index[-1]
    future_df     = model.make_future_dataframe(periods=future_days, freq="B")  # Business days
    full_forecast = model.predict(future_df)

    # Separate future from history
    future_forecast = full_forecast[full_forecast["ds"] > pd.Timestamp(last_date.date())]

    # ── Cache results in session state ────────────────────────────────────────
    st.session_state.update({
        "df":              df,
        "train_df":        train_df,
        "test_df":         test_df,
        "split_date":      split_date,
        "test_forecast":   test_forecast,
        "full_forecast":   full_forecast,
        "future_forecast": future_forecast,
        "test_rmse":       test_rmse,
        "forecast_done":   True,
    })

# ── Pull from session state ───────────────────────────────────────────────────
df              = st.session_state["df"]
train_df        = st.session_state["train_df"]
test_df         = st.session_state["test_df"]
split_date      = st.session_state["split_date"]
test_forecast   = st.session_state["test_forecast"]
full_forecast   = st.session_state["full_forecast"]
future_forecast = st.session_state["future_forecast"]
test_rmse       = st.session_state["test_rmse"]

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Data Points",   f"{len(df):,}")
k2.metric("Training Days", f"{len(train_df):,}")
k3.metric("Test Days",     f"{len(test_df):,}")
k4.metric("Test RMSE",     f"${test_rmse:.2f}")

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Full Overview",
    "🔍 Train / Test Evaluation",
    "🔮 Future Forecast",
    "📊 Components",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Full Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("AAPL Closing Price — 5-Year History + Forecast")

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"],
        name="Historical Price",
        line=dict(color="#1f77b4", width=1.5),
    ))

    # Prophet fitted values (in-sample)
    in_sample = full_forecast[full_forecast["ds"] <= pd.Timestamp(df.index[-1].date())]
    fig.add_trace(go.Scatter(
        x=in_sample["ds"], y=in_sample["yhat"],
        name="Prophet Fit",
        line=dict(color="orange", width=1, dash="dot"),
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_forecast["ds"], y=future_forecast["yhat"],
        name="Forecast (next 1 yr)",
        line=dict(color="green", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_forecast["ds"].values, future_forecast["ds"].values[::-1]]),
        y=np.concatenate([future_forecast["yhat_upper"].values, future_forecast["yhat_lower"].values[::-1]]),
        fill="toself", fillcolor="rgba(0,200,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
    ))

    # Train/test divider
    fig.add_vline(
        x=str(split_date.date()), line_dash="dash", line_color="red",
        annotation_text="Train | Test", annotation_position="top right",
    )

    fig.update_layout(
        height=500, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling statistics
    st.subheader("Rolling Mean & Standard Deviation (30-day window)")
    roll = df["price"].rolling(30)
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig2.add_trace(go.Scatter(x=df.index, y=df["price"],  name="Price",    line=dict(color="#1f77b4")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=roll.mean(), name="30d Mean",  line=dict(color="orange")),  row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=roll.std(),  name="30d Std",   line=dict(color="red")),     row=2, col=1)
    fig2.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train / Test Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation on 1-Year Test Set")

    col_l, col_r = st.columns([3, 1])

    with col_l:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=train_df.index, y=train_df["price"],
            name="Train", line=dict(color="#1f77b4"),
        ))
        fig3.add_trace(go.Scatter(
            x=test_df.index, y=test_df["price"],
            name="Actual (Test)", line=dict(color="green", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=test_forecast["ds"], y=test_forecast["yhat"],
            name="Predicted (Test)", line=dict(color="red", dash="dash", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=np.concatenate([test_forecast["ds"].values, test_forecast["ds"].values[::-1]]),
            y=np.concatenate([test_forecast["yhat_upper"].values, test_forecast["yhat_lower"].values[::-1]]),
            fill="toself", fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))
        fig3.update_layout(
            height=420, hovermode="x unified",
            xaxis_title="Date", yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        # Residuals stats
        residuals = test_df["price"].values - test_forecast["yhat"].values
        st.markdown("### Test Metrics")
        st.metric("RMSE",  f"${test_rmse:.2f}")
        st.metric("MAE",   f"${np.mean(np.abs(residuals)):.2f}")
        st.metric("MAPE",  f"{np.mean(np.abs(residuals / test_df['price'].values)) * 100:.2f}%")
        st.metric("Max Error", f"${np.max(np.abs(residuals)):.2f}")

    # Residuals plot
    st.subheader("Residuals (Actual − Predicted)")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=test_df.index, y=residuals,
        marker_color=["red" if r < 0 else "green" for r in residuals],
        name="Residual",
    ))
    fig4.add_hline(y=0, line_dash="dash", line_color="black")
    fig4.update_layout(height=280, xaxis_title="Date", yaxis_title="Residual (USD)")
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Future Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("1-Year Forecast Beyond Latest Data")

    fig5 = go.Figure()
    # Last 1 year of history for context
    recent = df[df.index >= df.index[-1] - timedelta(days=365)]
    fig5.add_trace(go.Scatter(
        x=recent.index, y=recent["price"],
        name="Recent History", line=dict(color="#1f77b4"),
    ))
    fig5.add_trace(go.Scatter(
        x=future_forecast["ds"], y=future_forecast["yhat"],
        name="Forecast", line=dict(color="green", width=2),
    ))
    fig5.add_trace(go.Scatter(
        x=np.concatenate([future_forecast["ds"].values, future_forecast["ds"].values[::-1]]),
        y=np.concatenate([future_forecast["yhat_upper"].values, future_forecast["yhat_lower"].values[::-1]]),
        fill="toself", fillcolor="rgba(0,200,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))
    fig5.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Summary table
    st.subheader("Monthly Forecast Summary")
    future_monthly = future_forecast.set_index("ds").resample("MS").agg(
        {"yhat": "mean", "yhat_lower": "mean", "yhat_upper": "mean"}
    ).reset_index()
    future_monthly.columns = ["Month", "Avg Forecast ($)", "Lower CI ($)", "Upper CI ($)"]
    future_monthly["Month"] = future_monthly["Month"].dt.strftime("%b %Y")
    future_monthly = future_monthly.round(2)
    st.dataframe(future_monthly, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Prophet Components
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Prophet Decomposition: Trend & Seasonality")

    comp_df = full_forecast[["ds", "trend", "yearly", "weekly"]].copy() if "weekly" in full_forecast.columns else full_forecast[["ds", "trend", "yearly"]].copy()

    fig6 = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Trend", "Yearly Seasonality", "Weekly Seasonality"],
        shared_xaxes=False,
        vertical_spacing=0.1,
    )

    fig6.add_trace(go.Scatter(x=comp_df["ds"], y=comp_df["trend"],  line=dict(color="#1f77b4")), row=1, col=1)
    fig6.add_trace(go.Scatter(x=comp_df["ds"], y=comp_df["yearly"], line=dict(color="orange")),  row=2, col=1)

    if "weekly" in comp_df.columns:
        # Use just one week for clarity
        week_sample = comp_df.tail(7)
        fig6.add_trace(go.Bar(
            x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            y=week_sample["weekly"].values,
            marker_color="steelblue",
        ), row=3, col=1)
    else:
        fig6.add_annotation(text="Weekly data not available", row=3, col=1, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    fig6.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.caption("Data source: Yahoo Finance via yfinance · Model: Facebook Prophet · Built with Streamlit")
