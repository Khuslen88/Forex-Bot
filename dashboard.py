"""
dashboard.py — Streamlit dashboard for the Forex RL Bot.

Professional control center for monitoring and operating the trading bot:
  - Live signals from the best model per pair (auto-selected from registry)
  - Bot start/stop controls (manages live_trader.py via signals)
  - Account overview, open positions, trade history
  - Model performance comparison (DQN vs PPO, daily vs hourly)
  - Backtest equity curves

Run:
  streamlit run dashboard.py
"""

import os
import sys
import json
import signal
import datetime
import subprocess

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN, PPO
from src.broker.paper_client import PaperClient
from src.features.indicators import add_indicators, load_econ_features
from src.features.sentiment import add_sentiment_to_df, HAS_TEXTBLOB
from config.settings import FOREX_PAIRS, MODELS_PATH

# ── Constants ────────────────────────────────────────────────────────────────

ROOT          = os.path.dirname(os.path.abspath(__file__))
ECON_PATH     = os.path.join(ROOT, "data", "economic", "fred_data.csv")
REGISTRY_PATH = os.path.join(MODELS_PATH, "registry.json")
STATUS_FILE   = os.path.join(ROOT, "bot_runtime.json")
PID_FILE      = os.path.join(ROOT, "bot_runtime.pid")
LIVE_TRADER   = os.path.join(ROOT, "live_trader.py")

ACTION_NAMES  = {0: "FLAT", 1: "LONG", 2: "SHORT"}
ACTION_COLORS = {"FLAT": "#6B7280", "LONG": "#10B981", "SHORT": "#EF4444"}

# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forex RL Bot — Control Center",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6B7280; }
    .status-pill {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .pill-green  { background: #D1FAE5; color: #065F46; }
    .pill-amber  { background: #FEF3C7; color: #92400E; }
    .pill-red    { background: #FEE2E2; color: #991B1B; }
    .pill-gray   { background: #E5E7EB; color: #374151; }
    .signal-card {
        padding: 16px; border-radius: 8px; border-left: 4px solid;
        background: #F9FAFB; margin-bottom: 8px;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #111827;
        margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)


# ── Registry helpers ─────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"models": [], "best_by_pair": {}}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def best_model_for_pair(pair: str, registry: dict, prefer_daily_v1: bool = True):
    """Return the highest-Sharpe model for the pair.

    With prefer_daily_v1 (default), only consider 1d v1 models — these are
    the production set (1H underperforms; v2 env experiments hurt results).
    Set False to include v2 / 1H / ensemble in the search.
    """
    candidates = [m for m in registry.get("models", []) if m["pair"] == pair]
    if prefer_daily_v1:
        prod = [m for m in candidates
                if m.get("timeframe") == "1d"
                and m.get("version", "v1") == "v1"]
        if prod:
            candidates = prod
    if not candidates:
        return None
    return max(candidates, key=lambda m: m["sharpe_ratio"])


# ── Bot runtime helpers ──────────────────────────────────────────────────────

def read_bot_status():
    if not os.path.exists(STATUS_FILE):
        return None
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def read_bot_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def is_bot_alive() -> bool:
    pid = read_bot_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_bot(pair: str, interval: int, dry_run: bool, units: int) -> str:
    """Spawn live_trader.py in --loop mode as a background process."""
    if is_bot_alive():
        return "Bot is already running."

    venv_python = os.path.join(ROOT, "venv", "bin", "python")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [
        python_bin, LIVE_TRADER,
        "--pair", pair,
        "--broker", "paper",
        "--loop",
        "--interval", str(interval),
        "--units", str(units),
    ]
    if dry_run:
        cmd.append("--dry-run")

    log_path = os.path.join(ROOT, "logs", "live_trader.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a")
    subprocess.Popen(
        cmd,
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=ROOT,
        start_new_session=True,
    )
    return "Bot started."


def stop_bot() -> str:
    pid = read_bot_pid()
    if pid is None:
        return "No PID file — bot not running."
    try:
        os.kill(pid, signal.SIGTERM)
        return f"Sent stop signal to PID {pid}."
    except ProcessLookupError:
        return "Bot already stopped (process not found)."
    except Exception as e:
        return f"Error stopping bot: {e}"


# ── Data + model loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_model_for_pair(pair: str, agent: str, timeframe: str):
    """Load a specific model. Returns (model, expected_obs_dim) or (None, 0)."""
    suffix = "" if timeframe == "1d" else f"_{timeframe}"
    fname = f"{agent.lower()}_feature_{pair}{suffix}.zip"
    path = os.path.join(MODELS_PATH, fname)
    if not os.path.exists(path):
        return None, 0
    try:
        loader = DQN if agent.upper() == "DQN" else PPO
        model = loader.load(path)
        return model, int(model.observation_space.shape[0])
    except Exception as e:
        st.error(f"Failed to load {fname}: {e}")
        return None, 0


@st.cache_data(ttl=300)
def fetch_pair_data(pair: str):
    """Fetch latest daily candles + indicators + econ + sentiment for a pair."""
    client = PaperClient()
    try:
        df = client.fetch_candles(pair, count=300, granularity="D")
        df = add_indicators(df)
        if os.path.exists(ECON_PATH):
            df = load_econ_features(ECON_PATH, df)
        df = add_sentiment_to_df(df, pair, live=HAS_TEXTBLOB)
        return df
    except Exception as e:
        st.warning(f"Could not fetch data for {pair}: {e}")
        return None


def build_observation(df: pd.DataFrame, expected_dim: int) -> np.ndarray:
    """Build a feature vector matching the trained model's observation shape."""
    window = 20
    prices = df["Close"].values.astype(np.float64)
    idx = len(prices) - 1

    window_prices = prices[idx - window: idx + 1]
    returns = np.diff(window_prices) / window_prices[:-1]
    rolling_vol = float(np.std(returns) * np.sqrt(252))
    base_obs = np.append(returns, [rolling_vol, 0.0]).astype(np.float32)

    close = float(prices[idx]) + 1e-8
    extra = [
        float(df["RSI"].iloc[-1]) / 100.0,
        float(df["MACD"].iloc[-1]) / close,
        float(df["MACD_signal"].iloc[-1]) / close,
        float(df["BB_width"].iloc[-1]),
        float(df["ATR"].iloc[-1]) / close,
        float(df["ADX"].iloc[-1]) / 100.0,
        float(df["SMA_ratio"].iloc[-1]) - 1.0,
    ]
    if "rate_diff" in df.columns:
        extra += [float(df["rate_diff"].iloc[-1]),
                  float(df["yield_curve"].iloc[-1]),
                  float(df["rate_diff_ma"].iloc[-1])]
    if "sentiment_score" in df.columns:
        extra += [float(df["sentiment_score"].iloc[-1]),
                  float(df["sentiment_vol"].iloc[-1]),
                  float(df["news_volume"].iloc[-1])]

    obs = np.concatenate([base_obs, np.array(extra, dtype=np.float32)])
    if len(obs) < expected_dim:
        obs = np.concatenate([obs, np.zeros(expected_dim - len(obs), dtype=np.float32)])
    elif len(obs) > expected_dim:
        obs = obs[:expected_dim]
    return obs


def get_signal(model, df: pd.DataFrame, expected_dim: int) -> int:
    obs = build_observation(df, expected_dim)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Forex RL Bot")
st.sidebar.caption("Capstone — Spring 2026")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if auto_refresh:
    st_autorefresh(interval=30_000, key="auto_refresh")

if st.sidebar.button("Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Bot Control**")

bot_alive = is_bot_alive()
bot_status = read_bot_status() or {}

if bot_alive:
    pill_class = {
        "running":  "pill-green",
        "sleeping": "pill-amber",
        "stopped":  "pill-gray",
    }.get(bot_status.get("state", ""), "pill-gray")
    state_label = bot_status.get("state", "unknown").upper()
    st.sidebar.markdown(
        f'<div class="status-pill {pill_class}">{state_label}</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="status-pill pill-gray">OFFLINE</div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

bot_pair = st.sidebar.selectbox("Pair to trade", ["all"] + FOREX_PAIRS, index=0)
bot_units = st.sidebar.number_input("Units per trade", value=1000, step=100, min_value=100)
bot_interval = st.sidebar.select_slider(
    "Check interval",
    options=[60, 300, 900, 1800, 3600, 14400, 86400],
    value=3600,
    format_func=lambda s: {60: "1 min", 300: "5 min", 900: "15 min",
                            1800: "30 min", 3600: "1 hour", 14400: "4 hours",
                            86400: "24 hours"}[s],
)
bot_dry = st.sidebar.checkbox("Dry run (signals only, no trades)", value=True)

ctrl_a, ctrl_b = st.sidebar.columns(2)
if ctrl_a.button("Start", type="primary", use_container_width=True, disabled=bot_alive):
    msg = start_bot(bot_pair, bot_interval, bot_dry, bot_units)
    st.sidebar.success(msg)
    st.rerun()
if ctrl_b.button("Stop", use_container_width=True, disabled=not bot_alive):
    msg = stop_bot()
    st.sidebar.info(msg)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Paper Account**")
if st.sidebar.button("Reset to $100k", use_container_width=True):
    PaperClient().reset_account()
    st.sidebar.success("Account reset.")
    st.cache_data.clear()
    st.rerun()


# ── Header + hero metrics ────────────────────────────────────────────────────

st.title("Forex RL Trading Bot")
st.caption("Reinforcement Learning trading agents — DQN + PPO across 5 forex pairs")

client = PaperClient()
acct = client.get_account_summary()
history = client.state.get("trade_history", [])

total_pnl = acct["balance"] - 100_000.0
pnl_pct = (total_pnl / 100_000.0) * 100

if history:
    pnls = [t["pl"] for t in history if "pl" in t]
    win_rate = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
else:
    win_rate = 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Account Balance", f"${acct['balance']:,.2f}")
m2.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
m3.metric("Unrealized P&L", f"${acct['unrealized_pnl']:,.2f}")
m4.metric("Open Positions", acct["open_trades"])
m5.metric("Closed Trades", len(history), f"{win_rate:.1f}% wins")

status = read_bot_status()
if status and bot_alive:
    state = status.get("state", "unknown")
    pair_run = status.get("pair", "?")
    interval_s = status.get("interval", 0)
    next_run = status.get("next_cycle_at", status.get("cycle_started_at", ""))
    if state == "sleeping":
        nrt = (datetime.datetime.fromisoformat(next_run)
               if next_run else None)
        next_str = nrt.strftime("%H:%M:%S") if nrt else "?"
        st.info(f"**Bot {state}** — pair: `{pair_run}`  "
                f"·  interval: `{interval_s}s`  ·  next check at `{next_str}`")
    else:
        st.success(f"**Bot {state}** — pair: `{pair_run}`  "
                   f"·  interval: `{interval_s}s`")
elif status and not bot_alive:
    st.warning("Bot process is gone but status file exists. "
               "Use **Stop** then **Start** to clear.")

# ── Main tabs ────────────────────────────────────────────────────────────────

tab_overview, tab_perf, tab_history, tab_charts, tab_backtest = st.tabs([
    "Overview",
    "Model Performance",
    "Trade History",
    "Price Charts",
    "Backtest Reports",
])


# ── Tab 1: Overview ──────────────────────────────────────────────────────────

with tab_overview:
    registry = load_registry()

    st.markdown('<div class="section-header">Recommended Model Per Pair</div>',
                unsafe_allow_html=True)
    st.caption("Auto-selected by highest Sharpe ratio across DQN/PPO and 1d/1h timeframes.")

    rec_rows = []
    for pair in FOREX_PAIRS:
        best = best_model_for_pair(pair, registry)
        if best is None:
            rec_rows.append({"Pair": pair, "Best Agent": "—", "Timeframe": "—",
                             "Return": "—", "Sharpe": "—", "Win Rate": "—",
                             "Profit Factor": "—"})
        else:
            rec_rows.append({
                "Pair":          pair,
                "Best Agent":    best["agent"],
                "Timeframe":     best["timeframe"],
                "Return":        f"{best['total_return']*100:+.2f}%",
                "Sharpe":        f"{best['sharpe_ratio']:+.3f}",
                "Win Rate":      f"{best['win_rate']*100:.1f}%",
                "Profit Factor": f"{best['profit_factor']:.2f}",
            })
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Live Signals (Best Model Per Pair)</div>',
                unsafe_allow_html=True)

    sig_cols = st.columns(len(FOREX_PAIRS))
    for i, pair in enumerate(FOREX_PAIRS):
        with sig_cols[i]:
            best = best_model_for_pair(pair, registry)
            if best is None:
                st.warning(f"**{pair}**\nNo trained model")
                continue
            model, dim = load_model_for_pair(pair, best["agent"], best["timeframe"])
            if model is None:
                st.warning(f"**{pair}**\nLoad failed")
                continue
            df = fetch_pair_data(pair)
            if df is None or len(df) < 21:
                st.warning(f"**{pair}**\nNo data")
                continue
            action = get_signal(model, df, dim)
            sig_name = ACTION_NAMES[action]
            color = ACTION_COLORS[sig_name]
            price = df["Close"].iloc[-1]
            daily_chg = ((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100

            st.markdown(
                f"""<div class="signal-card" style="border-color: {color};">
                    <div style="font-size: 0.85rem; color: #6B7280;">{pair}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
                        {sig_name}
                    </div>
                    <div style="font-size: 0.8rem; color: #374151; margin-top: 4px;">
                        Price: <b>{price:.5f}</b><br>
                        Day: {daily_chg:+.2f}%<br>
                        Model: {best['agent']} · {best['timeframe']}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">Execute Trade</div>',
                unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns([2, 1, 1])
    with ec1:
        exec_pair = st.selectbox("Pair", FOREX_PAIRS, key="exec_pair")
    with ec2:
        exec_units = st.number_input("Units", value=1000, step=100, key="exec_units")
    with ec3:
        exec_dry = st.checkbox("Dry run", value=True, key="exec_dry")

    if st.button("Run Decision Now", type="primary"):
        best = best_model_for_pair(exec_pair, registry)
        if best is None:
            st.error(f"No model for {exec_pair}")
        else:
            model, dim = load_model_for_pair(exec_pair, best["agent"], best["timeframe"])
            df = fetch_pair_data(exec_pair)
            if model and df is not None:
                action = get_signal(model, df, dim)
                sig_name = ACTION_NAMES[action]
                price = df["Close"].iloc[-1]
                st.info(f"Signal: **{sig_name}** at {price:.5f}  "
                        f"(model: {best['agent']} {best['timeframe']})")

                if exec_dry:
                    st.warning("Dry run — no trade executed.")
                else:
                    cur = client.get_net_position(exec_pair)
                    target = {0: 0, 1: 1, 2: -1}[action]
                    if target == cur:
                        st.info("No change — holding current position.")
                    else:
                        if cur != 0:
                            client.close_position(exec_pair)
                            st.info(f"Closed prior {exec_pair} position.")
                        if target != 0:
                            order = exec_units if target == 1 else -exec_units
                            res = client.place_order(exec_pair, order)
                            st.success(f"Filled at {res['price']:.5f}  "
                                       f"({order:+d} units)")
                    st.cache_data.clear()
                    st.rerun()

    st.markdown('<div class="section-header">Open Positions</div>',
                unsafe_allow_html=True)
    positions = client.state.get("positions", {})
    if positions:
        rows = []
        for sym, pos in positions.items():
            cur_price = client._get_current_price(sym)
            units = pos["units"]
            entry = pos["entry_price"]
            direction = "LONG" if units > 0 else "SHORT"
            if cur_price:
                pnl = ((cur_price - entry) if units > 0 else (entry - cur_price)) * abs(units)
                pnl_pct = (((cur_price / entry) - 1) if units > 0
                           else ((entry / cur_price) - 1)) * 100
            else:
                pnl, pnl_pct = 0.0, 0.0
            rows.append({
                "Pair":      sym,
                "Direction": direction,
                "Units":     abs(units),
                "Entry":     f"{entry:.5f}",
                "Current":   f"{cur_price:.5f}" if cur_price else "—",
                "P&L":       f"${pnl:,.2f}",
                "P&L %":     f"{pnl_pct:+.2f}%",
                "Opened":    pos.get("opened_at", "")[:16],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")


# ── Tab 2: Model Performance ─────────────────────────────────────────────────

with tab_perf:
    registry = load_registry()
    if not registry.get("models"):
        st.warning("No model registry found. Run "
                   "`python src/utils/build_registry.py` to compute metrics.")
    else:
        st.caption(f"Registry last updated: {registry.get('updated_at', 'unknown')}")

        st.markdown('<div class="section-header">All Trained Models</div>',
                    unsafe_allow_html=True)

        df_models = pd.DataFrame(registry["models"])
        df_models["Return"]   = df_models["total_return"].apply(lambda x: f"{x*100:+.2f}%")
        df_models["Sharpe"]   = df_models["sharpe_ratio"].apply(lambda x: f"{x:+.3f}")
        df_models["Win Rate"] = df_models["win_rate"].apply(lambda x: f"{x*100:.1f}%")
        df_models["Drawdown"] = df_models["max_drawdown"].apply(lambda x: f"{x*100:+.1f}%")
        df_models["PF"]       = df_models["profit_factor"].apply(lambda x: f"{x:.2f}")
        df_models["Trades"]   = df_models["n_trades"]

        display = df_models[["pair", "agent", "timeframe", "Return", "Sharpe",
                             "Win Rate", "Drawdown", "PF", "Trades"]]
        display.columns = ["Pair", "Agent", "TF", "Return", "Sharpe",
                           "Win Rate", "Max DD", "Profit Factor", "Trades"]
        display = display.sort_values(["Pair", "TF", "Agent"]).reset_index(drop=True)
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Sharpe Ratio Comparison</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()
        for tf in ["1d", "1h"]:
            for agent in ["DQN", "PPO"]:
                rows = [m for m in registry["models"]
                        if m["timeframe"] == tf and m["agent"] == agent]
                rows.sort(key=lambda m: m["pair"])
                fig.add_trace(go.Bar(
                    name=f"{agent} {tf}",
                    x=[m["pair"] for m in rows],
                    y=[m["sharpe_ratio"] for m in rows],
                ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                      annotation_text="Target (Sharpe ≥ 1.0)",
                      annotation_position="top right")
        fig.update_layout(
            barmode="group",
            height=400,
            yaxis_title="Sharpe Ratio",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Total Return Comparison</div>',
                    unsafe_allow_html=True)

        fig2 = go.Figure()
        for tf in ["1d", "1h"]:
            for agent in ["DQN", "PPO"]:
                rows = [m for m in registry["models"]
                        if m["timeframe"] == tf and m["agent"] == agent]
                rows.sort(key=lambda m: m["pair"])
                fig2.add_trace(go.Bar(
                    name=f"{agent} {tf}",
                    x=[m["pair"] for m in rows],
                    y=[m["total_return"] * 100 for m in rows],
                ))
        fig2.update_layout(
            barmode="group",
            height=400,
            yaxis_title="Total Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Trade History ─────────────────────────────────────────────────────

with tab_history:
    if history:
        hist_df = pd.DataFrame(history)
        total = len(hist_df)
        winning = int((hist_df["pl"] > 0).sum()) if "pl" in hist_df else 0
        losing = total - winning
        win_rate_h = winning / total * 100 if total else 0
        total_pl = float(hist_df["pl"].sum()) if "pl" in hist_df else 0

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Trades", total)
        h2.metric("Win Rate", f"{win_rate_h:.1f}%")
        h3.metric("Winning / Losing", f"{winning} / {losing}")
        h4.metric("Net P&L", f"${total_pl:,.2f}")

        st.markdown('<div class="section-header">Trade Log</div>',
                    unsafe_allow_html=True)

        disp = hist_df.copy()
        disp["Direction"] = disp["units"].apply(lambda x: "LONG" if x > 0 else "SHORT")
        disp["Units"]     = disp["units"].abs()
        disp["P&L"]       = disp["pl"].apply(lambda x: f"${x:,.2f}")
        disp["Closed"]    = disp["closed_at"].astype(str).str[:16]
        disp = disp[["symbol", "Direction", "Units", "entry_price",
                     "exit_price", "P&L", "Closed"]]
        disp.columns = ["Pair", "Direction", "Units", "Entry", "Exit",
                        "P&L", "Closed At"]
        st.dataframe(disp.iloc[::-1], use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Cumulative P&L</div>',
                    unsafe_allow_html=True)
        cum_pl = hist_df["pl"].cumsum()
        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(
            y=cum_pl.values,
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
        ))
        fig_pl.update_layout(
            height=320,
            yaxis_title="Cumulative P&L ($)",
            xaxis_title="Trade #",
            margin=dict(t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_pl, use_container_width=True)
    else:
        st.info("No trades yet. Use the **Overview** tab or start the bot to "
                "generate trades.")


# ── Tab 4: Price Charts ──────────────────────────────────────────────────────

with tab_charts:
    chart_pair = st.selectbox("Pair", FOREX_PAIRS, key="chart_pair_sel")
    df_chart = fetch_pair_data(chart_pair)
    if df_chart is not None and len(df_chart) > 0:
        df_plot = df_chart.tail(90).copy()

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f"{chart_pair} Price", "RSI", "MACD"),
            row_heights=[0.55, 0.22, 0.23],
            shared_xaxes=True,
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Candlestick(
            x=df_plot["Date"],
            open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"], close=df_plot["Close"],
            name="Price", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["BB_upper"],
            line=dict(color="rgba(150,150,150,0.4)", dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["BB_lower"],
            line=dict(color="rgba(150,150,150,0.4)", dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["SMA_50"],
            line=dict(color="#F59E0B", width=1), name="SMA 50",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["RSI"],
            line=dict(color="#8B5CF6", width=1.5), showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      opacity=0.4, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      opacity=0.4, row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["MACD"],
            line=dict(color="#3B82F6", width=1.5), name="MACD",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["MACD_signal"],
            line=dict(color="#EF4444", width=1), name="Signal",
        ), row=3, col=1)
        fig.add_trace(go.Bar(
            x=df_plot["Date"], y=df_plot["MACD_hist"],
            marker_color=df_plot["MACD_hist"].apply(
                lambda x: "#10B981" if x > 0 else "#EF4444"
            ),
            showlegend=False,
        ), row=3, col=1)
        fig.update_layout(
            height=720,
            xaxis_rangeslider_visible=False,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Could not load chart data for {chart_pair}")


# ── Tab 5: Backtest Reports ──────────────────────────────────────────────────

with tab_backtest:
    available = []
    for pair in FOREX_PAIRS:
        for tf in ["", "_1h"]:
            path = os.path.join(ROOT, f"results_{pair}{tf}.html")
            if os.path.exists(path):
                label = f"{pair} ({tf[1:] if tf else 'daily'})"
                available.append((label, path))

    if not available:
        st.info("No backtest reports yet. Run "
                "`python demo.py --pair EURUSD` to generate one.")
    else:
        labels = [a[0] for a in available]
        sel = st.selectbox("Report", labels)
        path = dict(available)[sel]
        with open(path) as f:
            html = f.read()
        st.components.v1.html(html, height=720, scrolling=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Forex RL Bot · Capstone Spring 2026 · "
    f"Last refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Models indexed: {len(load_registry().get('models', []))}/20"
)
