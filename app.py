
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

st.set_page_config(
    page_title="AI Options Trading Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI OPTIONS TRADING ASSISTANT")
st.caption("SENSEX & NIFTY | Manual Market Data | BUY / WAIT / EXIT")

st.warning(
    "MANUAL DATA MODE: This application does not connect to any broker "
    "or exchange. Enter accurate completed candle data. Signals are "
    "rule-based decision support, not guaranteed predictions."
)

def read_candles(raw, name):
    try:
        df = pd.read_csv(StringIO(raw.strip()))
        df.columns = [str(c).strip().lower() for c in df.columns]

        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            raise ValueError("Required columns: open, high, low, close")

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 0
        else:
            df["volume"] = pd.to_numeric(
                df["volume"], errors="coerce"
            ).fillna(0)

        df = df.dropna(subset=required).reset_index(drop=True)

        if len(df) < 22:
            raise ValueError("Please enter at least 22 completed candles.")

        if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
            raise ValueError("Invalid candle high value.")

        if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("Invalid candle low value.")

        return df, None

    except Exception as e:
        return None, name + ": " + str(e)


def calculate_indicators(df):
    x = df.copy()

    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()

    change = x["close"].diff()
    gain = change.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()
    loss = (-change.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    previous = x["close"].shift(1)

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - previous).abs(),
        (x["low"] - previous).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.rolling(14).mean()
    x["volume_avg"] = x["volume"].rolling(20).mean()

    return x


def analyse(option_df, index_df, option_type):
    option = calculate_indicators(option_df)
    index = calculate_indicators(index_df)

    last = option.iloc[-1]
    previous = option.iloc[-2]
    idx = index.iloc[-1]

    score = 0
    reasons = []

    # Option premium trend
    if last["close"] > last["ema9"] > last["ema21"]:
        score += 2
        reasons.append("Option premium trend is bullish.")
    elif last["close"] < last["ema9"] < last["ema21"]:
        score -= 2
        reasons.append("Option premium trend is bearish.")
    else:
        reasons.append("Option premium trend is unclear.")

    # RSI momentum
    if 52 <= last["rsi"] <= 72:
        score += 1
        reasons.append("RSI momentum is positive.")
    elif last["rsi"] > 78:
        score -= 1
        reasons.append("RSI is overextended. Avoid chasing.")
    elif last["rsi"] < 42:
        score -= 1
        reasons.append("RSI momentum is weak.")
    else:
        reasons.append("RSI is neutral.")

    # Candle breakout
    if last["close"] > previous["high"]:
        score += 1
        reasons.append("Latest candle closed above previous high.")
    else:
        reasons.append("No bullish candle breakout confirmed.")

    # Volume confirmation
    if (
        last["volume"] > 0
        and pd.notna(last["volume_avg"])
        and last["volume"] >= last["volume_avg"]
    ):
        score += 1
        reasons.append("Volume is above its 20-candle average.")
    else:
        reasons.append("Volume confirmation is missing.")

    # Underlying index confirmation
    if option_type == "CE":
        if idx["close"] > idx["ema9"] > idx["ema21"]:
            score += 2
            reasons.append("Underlying index confirms bullish direction.")
        elif idx["close"] < idx["ema9"] < idx["ema21"]:
            score -= 2
            reasons.append("Underlying index opposes a CE purchase.")
        else:
            reasons.append("Underlying index trend is mixed.")

    else:
        if idx["close"] < idx["ema9"] < idx["ema21"]:
            score += 2
            reasons.append("Underlying index confirms bearish direction.")
        elif idx["close"] > idx["ema9"] > idx["ema21"]:
            score -= 2
            reasons.append("Underlying index opposes a PE purchase.")
        else:
            reasons.append("Underlying index trend is mixed.")

    return option, index, score, reasons


def rupees(value):
    return "₹{:,.2f}".format(value)


st.sidebar.header("CONTRACT DETAILS")

index_name = st.sidebar.selectbox(
    "Select Index", ["SENSEX", "NIFTY"]
)

contract = st.sidebar.text_input(
    "Contract Name",
    "SENSEX 19 NOV 2026 PE 80100"
)

option_type = st.sidebar.selectbox(
    "Option Type", ["CE", "PE"]
)

timeframe = st.sidebar.selectbox(
    "Candle Timeframe",
    ["1 Minute", "3 Minutes", "5 Minutes", "15 Minutes"]
)

st.sidebar.divider()

holding = st.sidebar.checkbox(
    "I already hold this option"
)

entry_price = st.sidebar.number_input(
    "My purchase price (₹)",
    min_value=0.0,
    value=0.0,
    step=0.05,
    disabled=not holding
)

st.header("1. OPTION PREMIUM CANDLES")

st.write(
    "Paste at least 22 completed candles. "
    "Enter oldest candle first."
)

option_csv = st.text_area(
    "Option CSV",
    height=180,
    placeholder="""time,open,high,low,close,volume
09:15,100,104,99,103,1200
09:16,103,105,102,104,1500
"""
)

st.header("2. UNDERLYING INDEX CANDLES")

st.write(
    "Enter SENSEX or NIFTY index candles using the same timeframe."
)

index_csv = st.text_area(
    "Index CSV",
    height=180,
    placeholder="""time,open,high,low,close,volume
09:15,80000,80030,79980,80020,1000
09:16,80020,80050,80010,80040,1100
"""
)

st.header("3. CURRENT MARKET PRICE")

c1, c2 = st.columns(2)

with c1:
    ltp = st.number_input(
        "Current Option LTP (₹)",
        min_value=0.0,
        value=0.0,
        step=0.05
    )

with c2:
    oi = st.number_input(
        "Open Interest (optional)",
        min_value=0,
        value=0
    )

analyse_button = st.button(
    "🔍 ANALYSE MARKET",
    type="primary",
    use_container_width=True
)

if analyse_button:

    if not contract.strip():
        st.error("Enter the contract name.")

    elif ltp <= 0:
        st.error("Enter the current option LTP.")

    elif holding and entry_price <= 0:
        st.error("Enter your actual purchase price.")

    else:
        option_df, error = read_candles(
            option_csv, "Option candles"
        )

        if error:
            st.error(error)

        else:
            index_df, error = read_candles(
                index_csv, "Underlying index candles"
            )

            if error:
                st.error(error)

            else:
                option, index, score, reasons = analyse(
                    option_df, index_df, option_type
                )

                last = option.iloc[-1]

                atr = last["atr"]

                if pd.isna(atr) or atr <= 0:
                    atr = last["high"] - last["low"]

                if atr <= 0:
                    atr = ltp * 0.02

                # Use actual purchase price for an open position.
                reference_entry = (
                    entry_price if holding else ltp
                )

                stop = max(
                    0.05,
                    reference_entry - atr
                )

                risk = max(reference_entry - stop, 0.05)

                target1 = reference_entry + 1.5 * risk
                target2 = reference_entry + 2.0 * risk

                if score >= 4:
                    signal = "BUY " + option_type
                elif score <= -1:
                    signal = "WAIT - SETUP NOT CONFIRMED"
                else:
                    signal = "WAIT - NEED MORE CONFIRMATION"

                st.divider()
                st.header("4. AI ANALYSIS RESULT")

                st.subheader(contract)

                if score >= 4:
                    st.success("SIGNAL: " + signal)
                elif score <= -1:
                    st.error("SIGNAL: " + signal)
                else:
                    st.warning("SIGNAL: " + signal)

                a, b, c, d = st.columns(4)

                a.metric(
                    "Current LTP",
                    rupees(ltp)
                )

                b.metric(
                    "RSI",
                    "{:.1f}".format(last["rsi"])
                )

                c.metric(
                    "Analysis Score",
                    str(score)
                )

                d.metric(
                    "OI Entered",
                    "{:,}".format(oi)
                )

                st.header("5. TRADE PLAN")

                a, b, c, d = st.columns(4)

                a.metric(
                    "Reference Entry",
                    rupees(reference_entry)
                )

                b.metric(
                    "Stop Loss",
                    rupees(stop)
                )

                c.metric(
                    "Target 1",
                    rupees(target1)
                )

                d.metric(
                    "Target 2",
                    rupees(target2)
                )

                st.info(
                    "These are indicative ATR-based planning levels. "
                    "They are not guaranteed execution prices. "
                    "Check liquidity, bid/ask spread and slippage."
                )

                if holding:
                    pnl = ltp - entry_price
                    pnl_percent = pnl / entry_price * 100

                    st.header("6. EXISTING POSITION REVIEW")

                    st.metric(
                        "Unrealised P/L per unit",
                        rupees(pnl),
                        delta="{:.2f}%".format(pnl_percent)
                    )

                    if ltp <= stop:
                        st.error(
                            "EXIT REVIEW: Stop-loss level reached."
                        )

                    elif ltp >= target2:
                        st.success(
                            "EXIT / BOOK PROFIT REVIEW: Target 2 reached."
                        )

                    elif ltp >= target1:
                        st.success(
                            "PARTIAL PROFIT REVIEW: Target 1 reached."
                        )

                    else:
                        st.info(
                            "HOLD REVIEW: Continue monitoring completed candles."
                        )

                st.header("7. WHY THIS SIGNAL?")

                for reason in reasons:
                    st.write("• " + reason)

                st.write(
                    "Option candles analysed:",
                    len(option_df)
                )

                st.write(
                    "Index candles analysed:",
                    len(index_df)
                )

                st.caption(
                    "Analysis time: " +
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                )

                st.warning(
                    "This is a rule-based prototype, not a trained "
                    "predictive AI. Backtest and paper-trade before "
                    "using real money."
                )


st.divider()

st.header("HOW TO USE")

st.markdown("""
1. Select SENSEX or NIFTY.
2. Enter your exact option contract.
3. Select CE or PE.
4. Copy completed option candles from your chart.
5. Copy corresponding index candles.
6. Enter the current option LTP.
7. Click ANALYSE MARKET.
8. Review BUY / WAIT, stop-loss and targets.

For an existing position, tick the holding checkbox and enter
your actual purchase price.

**Never use stale, incomplete or invented market data.**
""")
