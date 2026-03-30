"""
dashboard.py — Streamlit dashboard for the Forex RL Bot.

Shows real-time account status, positions, trade history, and model signals.

Run:
  streamlit run dashboard.py
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN
from src.broker.paper_client import PaperClient
from src.features.indicators import add_indicators, load_econ_features
from src.features.sentiment import add_sentiment_to_df, fetch_live_sentiment, HAS_TEXTBLOB
from config.settings import FOREX_PAIRS, MODELS_PATH

ECON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "economic", "fred_data.csv")

ACTION_NAMES = {0: "FLAT", 1: "LONG", 2: "SHORT"}
ACTION_COLORS = {"FLAT": "#9E9E9E", "LONG": "#4CAF50", "SHORT": "#F44336"}

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forex RL Bot",
    page_icon="📈",
    layout="wide",
)

# ── Helper functions ─────────────────────────────────────────────────────────


@st.cache_resource
def load_model(pair: str):
    """Load a trained DQN model."""
    path = os.path.join(MODELS_PATH, f"dqn_feature_{pair}.zip")
    if not os.path.exists(path):
        return None
    return DQN.load(path)


def get_model_signal(model, df: pd.DataFrame) -> tuple:
    """Get the model's current action from the latest data."""
    expected_dim = model.observation_space.shape[0]

    window = 20
    prices = df["Close"].values.astype(np.float64)
    idx = len(prices) - 1

    window_prices = prices[idx - window: idx + 1]
    returns = np.diff(window_prices) / window_prices[:-1]
    rolling_vol = float(np.std(returns) * np.sqrt(252))
    position = 0.0

    base_obs = np.append(returns, [rolling_vol, position]).astype(np.float32)

    close = float(prices[idx]) + 1e-8
    rsi = float(df["RSI"].iloc[-1]) / 100.0
    macd = float(df["MACD"].iloc[-1]) / close
    macd_sig = float(df["MACD_signal"].iloc[-1]) / close
    bb_width = float(df["BB_width"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) / close
    adx = float(df["ADX"].iloc[-1]) / 100.0
    sma_ratio = float(df["SMA_ratio"].iloc[-1]) - 1.0

    extra = [rsi, macd, macd_sig, bb_width, atr, adx, sma_ratio]

    if "rate_diff" in df.columns:
        extra.append(float(df["rate_diff"].iloc[-1]))
        extra.append(float(df["yield_curve"].iloc[-1]))
        extra.append(float(df["rate_diff_ma"].iloc[-1]))

    # Only include sentiment if the model was trained with it
    if "sentiment_score" in df.columns and (22 + len(extra) + 3) <= expected_dim:
        extra.append(float(df["sentiment_score"].iloc[-1]))
        extra.append(float(df["sentiment_vol"].iloc[-1]))
        extra.append(float(df["news_volume"].iloc[-1]))

    extra = np.array(extra, dtype=np.float32)
    obs = np.concatenate([base_obs, extra])

    # Pad or trim to match model's expected dimension
    if len(obs) < expected_dim:
        obs = np.concatenate([obs, np.zeros(expected_dim - len(obs), dtype=np.float32)])
    elif len(obs) > expected_dim:
        obs = obs[:expected_dim]

    action, _ = model.predict(obs, deterministic=True)
    return int(action), ACTION_NAMES[int(action)]


def fetch_pair_data(client: PaperClient, pair: str) -> pd.DataFrame:
    """Fetch candles, compute indicators, merge FRED data, and add sentiment."""
    try:
        df = client.fetch_candles(pair, count=300, granularity="D")
        df = add_indicators(df)
        # Merge FRED economic features
        if os.path.exists(ECON_PATH):
            df = load_econ_features(ECON_PATH, df)
        # Add sentiment features (live if textblob available, else proxy)
        df = add_sentiment_to_df(df, pair, live=HAS_TEXTBLOB)
        return df
    except Exception as e:
        st.warning(f"Could not fetch data for {pair}: {e}")
        return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Forex RL Bot")
st.sidebar.markdown("---")

# Account controls
if st.sidebar.button("Reset Account to $100k"):
    client = PaperClient()
    client.reset_account()
    st.sidebar.success("Account reset!")
    st.rerun()

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Trading Pairs**")
selected_pairs = st.sidebar.multiselect(
    "Select pairs to monitor",
    FOREX_PAIRS,
    default=FOREX_PAIRS,
)

st.sidebar.markdown("---")
trade_units = st.sidebar.number_input("Trade size (units)", value=1000, step=100)
dry_run = st.sidebar.checkbox("Dry run (no trades)", value=True)

# ── Main layout ──────────────────────────────────────────────────────────────

st.title("Forex RL Trading Bot — Dashboard")

# ── Account Overview ─────────────────────────────────────────────────────────

client = PaperClient()
acct = client.get_account_summary()

col1, col2, col3, col4 = st.columns(4)

total_pnl = acct["balance"] - 100_000.0
pnl_pct = (total_pnl / 100_000.0) * 100

col1.metric("Balance", f"${acct['balance']:,.2f}")
col2.metric("Unrealized P&L", f"${acct['unrealized_pnl']:,.2f}")
col3.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
col4.metric("Open Trades", acct["open_trades"])

st.markdown("---")

# ── Model Signals ────────────────────────────────────────────────────────────

st.subheader("Current Model Signals")

signal_cols = st.columns(len(selected_pairs))

for i, pair in enumerate(selected_pairs):
    with signal_cols[i]:
        model = load_model(pair)
        if model is None:
            st.warning(f"{pair}\nNo model")
            continue

        df = fetch_pair_data(client, pair)
        if df is None:
            continue

        action, signal_name = get_model_signal(model, df)
        current_price = df["Close"].iloc[-1]
        daily_change = ((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100

        # Signal card
        color = ACTION_COLORS[signal_name]
        st.markdown(
            f"""
            <div style="background-color: {color}20; border-left: 4px solid {color};
                        padding: 12px; border-radius: 4px; margin-bottom: 8px;">
                <h3 style="margin: 0; color: {color};">{pair}</h3>
                <p style="font-size: 24px; font-weight: bold; margin: 4px 0; color: {color};">
                    {signal_name}
                </p>
                <p style="margin: 0;">Price: {current_price:.5f}</p>
                <p style="margin: 0;">Daily: {daily_change:+.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Key indicators
        rsi_val = df["RSI"].iloc[-1]
        adx_val = df["ADX"].iloc[-1]
        sma_r = df["SMA_ratio"].iloc[-1]

        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        trend = "Trending" if adx_val > 25 else "Ranging"
        direction = "Uptrend" if sma_r > 1 else "Downtrend"

        st.caption(f"RSI: {rsi_val:.1f} ({rsi_status})")
        st.caption(f"ADX: {adx_val:.1f} ({trend})")
        st.caption(f"SMA Ratio: {sma_r:.4f} ({direction})")

st.markdown("---")

# ── Execute Trades ───────────────────────────────────────────────────────────

st.subheader("Execute Trades")

exec_col1, exec_col2 = st.columns([3, 1])

with exec_col1:
    trade_pair = st.selectbox("Pair", selected_pairs)

with exec_col2:
    st.write("")  # spacer
    st.write("")
    execute_btn = st.button("Run Trading Decision", type="primary")

if execute_btn:
    model = load_model(trade_pair)
    if model is None:
        st.error(f"No trained model for {trade_pair}")
    else:
        df = fetch_pair_data(client, trade_pair)
        if df is not None:
            action, signal_name = get_model_signal(model, df)
            current_price = df["Close"].iloc[-1]

            st.info(f"Model signal for {trade_pair}: **{signal_name}** at {current_price:.5f}")

            if dry_run:
                st.warning("Dry run mode — no trade placed. Uncheck 'Dry run' in sidebar to trade.")
            else:
                symbol = trade_pair
                current_pos = client.get_net_position(symbol)
                target_pos = {0: 0, 1: 1, 2: -1}[action]

                if target_pos == current_pos:
                    st.info("No change needed — holding current position.")
                else:
                    if current_pos != 0:
                        client.close_position(symbol)
                        st.info(f"Closed existing {trade_pair} position.")

                    if target_pos != 0:
                        order_units = trade_units if target_pos == 1 else -trade_units
                        result = client.place_order(symbol, order_units)
                        st.success(
                            f"Placed {'BUY' if target_pos == 1 else 'SELL'} order: "
                            f"{abs(order_units)} units at {result['price']:.5f}"
                        )
                    else:
                        st.info("Going flat — no new position.")

                st.rerun()

# Execute all pairs button
if st.button("Trade All Pairs"):
    if dry_run:
        st.warning("Dry run mode — no trades placed.")
    else:
        for pair in selected_pairs:
            model = load_model(pair)
            if model is None:
                continue
            df = fetch_pair_data(client, pair)
            if df is None:
                continue

            action, signal_name = get_model_signal(model, df)
            current_pos = client.get_net_position(pair)
            target_pos = {0: 0, 1: 1, 2: -1}[action]

            if target_pos != current_pos:
                if current_pos != 0:
                    client.close_position(pair)
                if target_pos != 0:
                    order_units = trade_units if target_pos == 1 else -trade_units
                    client.place_order(pair, order_units)
                    st.success(f"{pair}: {signal_name} — {order_units:+d} units")
                else:
                    st.info(f"{pair}: Going FLAT")
            else:
                st.info(f"{pair}: Holding {signal_name}")

        st.rerun()

st.markdown("---")

# ── Open Positions ───────────────────────────────────────────────────────────

st.subheader("Open Positions")

positions = client.state.get("positions", {})
if positions:
    pos_data = []
    for symbol, pos in positions.items():
        current = client._get_current_price(symbol)
        units = pos["units"]
        entry = pos["entry_price"]
        direction = "LONG" if units > 0 else "SHORT"

        if current:
            if units > 0:
                pnl = (current - entry) * abs(units)
            else:
                pnl = (entry - current) * abs(units)
            pnl_pct = ((current / entry) - 1) * 100 if units > 0 else ((entry / current) - 1) * 100
        else:
            pnl = 0.0
            pnl_pct = 0.0

        pos_data.append({
            "Pair": symbol,
            "Direction": direction,
            "Units": abs(units),
            "Entry": f"{entry:.5f}",
            "Current": f"{current:.5f}" if current else "N/A",
            "P&L": f"${pnl:,.2f}",
            "P&L %": f"{pnl_pct:+.2f}%",
            "Opened": pos.get("opened_at", "N/A")[:16],
        })

    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
else:
    st.info("No open positions.")

st.markdown("---")

# ── Trade History ────────────────────────────────────────────────────────────

st.subheader("Trade History")

history = client.state.get("trade_history", [])
if history:
    hist_df = pd.DataFrame(history)
    hist_df["direction"] = hist_df["units"].apply(lambda x: "LONG" if x > 0 else "SHORT")
    hist_df["pl_color"] = hist_df["pl"].apply(lambda x: "green" if x > 0 else "red")

    # Summary metrics
    total_trades = len(hist_df)
    winning = len(hist_df[hist_df["pl"] > 0])
    losing = len(hist_df[hist_df["pl"] <= 0])
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    total_pl = hist_df["pl"].sum()

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Total Trades", total_trades)
    hcol2.metric("Win Rate", f"{win_rate:.1f}%")
    hcol3.metric("Winning", winning)
    hcol4.metric("Total P&L", f"${total_pl:,.2f}")

    # Trade table
    display_df = hist_df[["symbol", "direction", "units", "entry_price", "exit_price", "pl", "closed_at"]].copy()
    display_df.columns = ["Pair", "Direction", "Units", "Entry", "Exit", "P&L", "Closed"]
    display_df["Units"] = display_df["Units"].abs()
    display_df["P&L"] = display_df["P&L"].apply(lambda x: f"${x:,.2f}")
    display_df["Closed"] = display_df["Closed"].str[:16]

    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)

    # P&L chart
    cumulative_pl = hist_df["pl"].cumsum()
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(
        y=cumulative_pl.values,
        mode="lines+markers",
        name="Cumulative P&L",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.1)",
    ))
    fig_pl.update_layout(
        title="Cumulative P&L",
        yaxis_title="P&L ($)",
        xaxis_title="Trade #",
        height=300,
    )
    st.plotly_chart(fig_pl, use_container_width=True)
else:
    st.info("No trade history yet. Execute some trades to see results here.")

st.markdown("---")

# ── Price Charts ─────────────────────────────────────────────────────────────

st.subheader("Price Charts (Last 60 Days)")

chart_pair = st.selectbox("Select pair for chart", selected_pairs, key="chart_pair")

df_chart = fetch_pair_data(client, chart_pair)
if df_chart is not None:
    # Take last 60 rows
    df_plot = df_chart.tail(60).copy()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f"{chart_pair} Price", "RSI", "MACD"),
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot["Date"],
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="Price",
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["BB_upper"],
        name="BB Upper", line=dict(color="rgba(150,150,150,0.3)", dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["BB_lower"],
        name="BB Lower", line=dict(color="rgba(150,150,150,0.3)", dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
    ), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["SMA_50"],
        name="SMA 50", line=dict(color="#FF9800", width=1),
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["RSI"],
        name="RSI", line=dict(color="#9C27B0", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["MACD"],
        name="MACD", line=dict(color="#2196F3", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["MACD_signal"],
        name="Signal", line=dict(color="#FF5722", width=1),
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=df_plot["Date"], y=df_plot["MACD_hist"],
        name="Histogram",
        marker_color=df_plot["MACD_hist"].apply(
            lambda x: "#4CAF50" if x > 0 else "#F44336"
        ),
    ), row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Backtest Results ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Backtest Results")

results_files = {pair: f"results_{pair}.html" for pair in FOREX_PAIRS}
available_results = {pair: path for pair, path in results_files.items()
                     if os.path.exists(os.path.join(os.path.dirname(__file__), path))}

if available_results:
    result_pair = st.selectbox("Select pair", list(available_results.keys()), key="result_pair")
    result_path = os.path.join(os.path.dirname(__file__), available_results[result_pair])

    with open(result_path, "r") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=750, scrolling=True)
else:
    st.info("No backtest results found. Run `python demo.py --pair EURUSD` first.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Forex RL Bot — Capstone Project | "
    f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Models: {len([p for p in FOREX_PAIRS if os.path.exists(os.path.join(MODELS_PATH, f'dqn_feature_{p}.zip'))])}/{len(FOREX_PAIRS)} pairs trained"
)
