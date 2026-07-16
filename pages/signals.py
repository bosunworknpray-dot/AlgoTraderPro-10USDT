# signals.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime
import asyncio
from ml import MLFilter
from sqlalchemy import update

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import TradeModel, db_manager
from indicators import get_candles
from signal_generator import generate_signals, get_usdt_symbols, analyze_single_symbol
from notifications import send_all_notifications
from engine import TradingEngine
from exceptions import APIException

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

# Apply black background theme
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1a1a;
    }
    .stButton>button {
        color: #000000;
        background-color: #00ff88;
        border-color: #00ff88;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>textarea, .stSlider>div>div>div>input {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff88 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1a1a;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        color: #ffffff;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff88;
        color: #000000;
    }
    .stDivider {
        border-color: #333333;
    }
    .stExpanderHeader {
        background-color: #1a1a1a;
        color: #ffffff;
        font-size: 18px;
    }
    .stExpanderContent {
        background-color: #0a0a0a;
        padding: 20px;
    }
    .stPlotlyChart, .stVegaLiteChart {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
    .stMarkdown a {
        color: #00ff88;
    }
    .stMetric {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333333;
    }
    .stDataFrame {
        background-color: #1a1a1a;
        border-radius: 10px;
    }
    .signal-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        border: 2px solid #333;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Signals - AlgoTraderPro", page_icon="target", layout="wide")

def create_signal_chart(signal_data):
    """Create a candlestick chart with entry, TP, SL, trail, and liquidation lines"""
    try:
        symbol = signal_data.get('symbol', signal_data.get('Symbol', 'BTCUSDT'))
        try:
            entry = float(signal_data.get('entry', signal_data.get('Entry', 0)) or 0)
            tp = float(signal_data.get('tp', signal_data.get('TP', 0)) or 0)
            sl = float(signal_data.get('sl', signal_data.get('SL', 0)) or 0)
            trail = float(signal_data.get('trail', signal_data.get('Trail', 0)) or 0)
            liquidation = float(signal_data.get('liquidation', signal_data.get('Liquidation', 0)) or 0)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid price values for {symbol}: {e}")
            return None

        candles = get_candles(symbol, "60", limit=60)
        if not candles or not isinstance(candles, list) or len(candles) == 0:
            logger.warning(f"No candlestick data for {symbol}")
            return None

        df = pd.DataFrame(candles)
        required_columns = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Invalid candlestick data format for {symbol}: missing columns")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isna().any():
            logger.error(f"Invalid timestamp data for {symbol}")
            return None

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))

        if entry > 0:
            fig.add_hline(y=entry, line_dash="dash", line_color="#00ccff",
                         annotation_text=f"Entry: ${entry:,.4f}")
        if tp > 0:
            fig.add_hline(y=tp, line_dash="dot", line_color="#00ff88",
                         annotation_text=f"Take Profit: ${tp:,.4f}")
        if sl > 0:
            fig.add_hline(y=sl, line_dash="dot", line_color="#ff4444",
                         annotation_text=f"Stop Loss: ${sl:,.4f}")
        if trail > 0:
            fig.add_hline(y=trail, line_dash="dashdot", line_color="#aa00ff",
                         annotation_text=f"Trail: ${trail:,.4f}")
        if liquidation > 0:
            fig.add_hline(y=liquidation, line_dash="dashdot", line_color="#ff8800",
                         annotation_text=f"Liquidation: ${liquidation:,.4f}")

        fig.update_layout(
            title=f"{symbol} - Multi-Timeframe Signal",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating signal chart for {symbol}: {e}")
        return None

def display_signal_details(signal):
    """Display detailed signal information"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Signal Details**")
        st.write(f"**Symbol:** {signal.get('symbol', 'N/A')}")
        side = "LONG" if str(signal.get('side', '')).upper() == "LONG" else "SHORT"
        st.write(f"**Side:** {side}")
        st.write(f"**Type:** {signal.get('type', 'N/A')}")
        st.write(f"**Score:** {signal.get('score', 0):.1f}%")
        st.write(f"**BB Slope:** {signal.get('bb_slope', 'N/A')}")

    with col2:
        st.markdown("**Price Levels**")
        market_price = signal.get('market', 0)
        try:
            market_price = float(market_price)
        except:
            market_price = 0
        st.write(f"**Market Price:** ${market_price:,.4f}")
        st.write(f"**Entry:** ${float(signal.get('entry', 0)):,.4f}")
        st.write(f"**Take Profit:** ${float(signal.get('tp', 0)):,.4f}")
        st.write(f"**Stop Loss:** ${float(signal.get('sl', 0)):,.4f}")
        st.write(f"**Trail:** ${float(signal.get('trail', 0)):,.4f}")

    with col3:
        st.markdown("**Risk Management**")
        st.write(f"**Liquidation:** ${float(signal.get('liquidation', 0)):,.4f}")
        st.write(f"**Margin USDT:** ${float(signal.get('margin_usdt', 0)):.2f}")
        st.write(f"**Leverage:** 20x")
        st.write(f"**Market Type:** {signal.get('market', 'N/A')}")
        st.write(f"**Generated:** {signal.get('time', 'N/A')}")

async def execute_real_trade(engine, signal):
    """Execute a real trade based on a signal"""
    try:
        trade_data = {
            "symbol": signal["symbol"],
            "side": "Buy" if signal["side"] == "LONG" else "Sell",
            "qty": 0.001,
            "entry": signal["entry"],
            "leverage": 20,
            "sl": signal["sl"],
            "tp": signal["tp"],
            "trail": signal["trail"],
            "liquidation": signal["liquidation"],
            "margin_usdt": signal["margin_usdt"],
            "margin_mode": "CROSS",
            "order_id": f"signal_{signal['symbol']}_{int(datetime.now().timestamp())}",
            "virtual": False,
            "status": "open",
            "strategy": "MultiTF"
        }

        success = db_manager.add_trade(trade_data)
        if not success:
            st.error("Failed to save trade to database")
            return False

        success = await engine.execute_real_trade([trade_data])
        if success:
            await asyncio.sleep(2)
            engine.sync_real_trades()
            engine.sync_real_balance()
            return True
        else:
            session = db_manager._get_session()
            session.execute(update(TradeModel).where(TradeModel.order_id == trade_data["order_id"]).values(status="failed"))
            session.commit()
            return False

    except Exception as e:
        logger.error(f"Real trade error: {e}")
        return False

def main():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; border-bottom: 3px solid #00ff88; margin-bottom: 3rem;">
        <h1 style="color: #00ff88; margin: 0; font-size: 3rem;">Multi-Timeframe Trading Signals</h1>
        <p style="color: #888; margin: 10px 0 0; font-size: 1.3rem;">AI-Powered • 15m + 60m + 240m Alignment • LONG & SHORT</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("How to Use Signals", expanded=False):
        st.markdown("""
        ### Signal Generation Process

        1. **Click "Generate New Signals"** to scan top 100 USDT perpetuals
        2. System analyzes 15m, 60m, and 240m timeframes simultaneously
        3. Only signals where **ALL timeframes agree** are shown
        4. Machine learning scores each opportunity (0-100%)
        5. Top high-conviction signals are displayed

        ### Understanding Signal Scores

        - **90-100%**: Extremely strong — highest probability
        - **70-89%**: Very strong — excellent setup
        - **50-69%**: Moderate — good risk/reward
        - **Below 50%**: Weak — avoid

        ### Executing Trades

        **Virtual Mode:** Safe paper trading  
        **Real Mode:** Live trading with real funds

        Each signal includes:
        - Entry price
        - Take Profit target
        - Stop Loss protection
        - Trailing stop
        - Liquidation price
        - Required margin
        """)

    st.divider()

    # Initialize session state
    if 'generated_signals' not in st.session_state:
        st.session_state.generated_signals = []
        try:
            db_signals = db_manager.get_signals(limit=10)
            st.session_state.generated_signals = [s.to_dict() for s in db_signals if s.score >= 50]
        except Exception as e:
            logger.error(f"Error loading initial signals: {e}")

    if 'signal_generation_in_progress' not in st.session_state:
        st.session_state.signal_generation_in_progress = False

    # Sidebar controls
    with st.sidebar:
        st.header("Signal Controls")

        trading_mode = st.selectbox(
            "Trading Mode",
            ["virtual", "real"],
            index=0 if st.session_state.get('trading_mode', 'virtual') == 'virtual' else 1,
            key="signals_trading_mode"
        )
        st.session_state.trading_mode = trading_mode

        st.divider()

        st.subheader("Generation Settings")
        top_n_signals = st.slider("Number of Signals", 1, 20, 10)
        min_score = st.slider("Minimum Score", 30, 100, 60)

        available_symbols = get_usdt_symbols(100)
        selected_symbols = st.multiselect(
            "Select Symbols (leave empty for auto)",
            available_symbols,
            default=[]
        )

        st.divider()

        if st.button("Generate New Signals", type="primary",
                    disabled=st.session_state.signal_generation_in_progress):
            st.session_state.signal_generation_in_progress = True

        if st.button("Send Notifications",
                    disabled=len(st.session_state.generated_signals) == 0):
            if st.session_state.generated_signals:
                try:
                    send_all_notifications(st.session_state.generated_signals)
                    st.success("Notifications sent!")
                except Exception as e:
                    st.error(f"Notification error: {e}")

        st.divider()

        st.subheader("Database Filters")
        symbol_filter = st.text_input("Symbol Filter", placeholder="BTC, ETH...")
        side_filter = st.selectbox("Side Filter", ["All", "LONG", "SHORT"])

    # Handle signal generation
    if st.session_state.signal_generation_in_progress:
        with st.spinner("Generating multi-timeframe signals..."):
            try:
                symbols_to_scan = selected_symbols if selected_symbols else get_usdt_symbols(100)
                signals = generate_signals(
                    symbols_to_scan,
                    interval="60",
                    top_n=top_n_signals * 2,
                    trading_mode=trading_mode
                )
                filtered_signals = [s for s in signals if s.get('score', 0) >= min_score]
                st.session_state.generated_signals = filtered_signals[:top_n_signals]
                st.success(f"Generated {len(filtered_signals)} aligned signals!")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Signal generation error: {e}")
            finally:
                st.session_state.signal_generation_in_progress = False
                st.rerun()

    # Initialize engine
    engine = st.session_state.get("engine")
    if not engine:
        try:
            st.session_state.engine = TradingEngine()
            engine = st.session_state.engine
        except Exception as e:
            st.error("Failed to initialize trading engine")
            st.stop()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["New Signals", "Database Signals", "Single Symbol Analysis", "ML Signal Filter"])

    with tab1:
        st.subheader("Latest Multi-Timeframe Signals")
        signals = st.session_state.generated_signals
        if signals:
            for idx, sig in enumerate(signals):
                with st.container():
                    st.markdown(f"<div class='signal-card'>", unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        side_emoji = "LONG" if sig.get('side') == 'LONG' else "SHORT"
                        st.markdown(f"### {sig.get('symbol')} — {side_emoji} — Score: {sig.get('score'):.1f}%")
                        st.write(f"**Entry:** ${sig.get('entry'):,.4f} → **TP:** ${sig.get('tp'):,.4f} → **SL:** ${sig.get('sl'):,.4f}")
                        st.write(f"**Trail:** ${sig.get('trail'):,.4f} | **Liq:** ${sig.get('liquidation'):,.4f} | **Margin:** ${sig.get('margin_usdt'):.2f}")
                        st.caption(f"Type: {sig.get('type')} • BB Slope: {sig.get('bb_slope')} • Time: {sig.get('time')}")
                    with col2:
                        if st.button("Virtual Trade", key=f"virt_{idx}"):
                            engine.execute_virtual_trade(sig)
                            st.success("Virtual trade executed!")
                        if trading_mode == "real" and st.button("Real Trade", key=f"real_{idx}"):
                            with st.spinner("Executing real trade..."):
                                loop = asyncio.new_event_loop()
                                success = loop.run_until_complete(execute_real_trade(engine, sig))
                                loop.close()
                                if success:
                                    st.success("Real trade executed!")
                                    st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                    chart = create_signal_chart(sig)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    st.markdown("---")
        else:
            st.info("Click 'Generate New Signals' to start scanning")

    with tab2:
        st.subheader("Historical Signals from Database")
        try:
            db_signals = db_manager.get_signals(limit=100)
            if db_signals:
                filtered = []
                for s in db_signals:
                    sig_dict = s.to_dict()
                    if symbol_filter and symbol_filter.upper() not in sig_dict.get('symbol', '').upper():
                        continue
                    if side_filter != "All" and sig_dict.get('side', '').upper() != side_filter:
                        continue
                    filtered.append(sig_dict)

                if filtered:
                    df = pd.DataFrame([{
                        "Symbol": s.get("symbol", "N/A"),
                        "Side": "LONG" if s.get("side", "").upper() == "LONG" else "SHORT",
                        "Score": f"{s.get('score', 0):.1f}%",
                        "Entry": f"${s.get('entry', 0):,.4f}",
                        "TP": f"${s.get('tp', 0):,.4f}",
                        "SL": f"${s.get('sl', 0):,.4f}",
                        "Margin": f"${s.get('margin_usdt', 0):.2f}",
                        "Time": str(s.get("created_at", ""))[:19]
                    } for s in filtered])
                    st.dataframe(df, use_container_width=True, height=600)
                else:
                    st.info("No signals match filters")
            else:
                st.info("No signals in database")
        except Exception as e:
            st.error(f"Database error: {e}")

    with tab3:
        st.subheader("Single Symbol Deep Analysis")
        col1, col2 = st.columns([1, 2])
        with col1:
            symbol = st.selectbox("Select Symbol", get_usdt_symbols(50))
            if st.button("Analyze Symbol"):
                with st.spinner("Running full multi-timeframe analysis..."):
                    result = analyze_single_symbol(symbol)
                    if result and result.get("score", 0) >= 30:
                        st.session_state.single_analysis = result
                        st.success("Strong signal detected!")
                    else:
                        st.warning("No strong alignment found")
        with col2:
            if "single_analysis" in st.session_state:
                s = st.session_state.single_analysis
                display_signal_details(s)
                chart = create_signal_chart(s)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Execute Virtual Trade"):
                        engine.execute_virtual_trade(s)
                        st.success("Virtual trade opened!")
                with c2:
                    if trading_mode == "real" and st.button("Execute Real Trade"):
                        with st.spinner("Sending real order..."):
                            loop = asyncio.new_event_loop()
                            success = loop.run_until_complete(execute_real_trade(engine, s))
                            loop.close()
                            st.success("Real trade executed!") if success else st.error("Failed")

    with tab4:
        st.subheader("ML-Powered Signal Filtering")
        try:
            ml_filter = MLFilter()
            threshold = st.slider("ML Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            all_signals = db_manager.get_signals(limit=200)
            
            if not all_signals:
                st.info("No signals available.")
            else:
                filtered = ml_filter.filter_signals(all_signals, threshold=threshold)
                st.write(f"**{len(filtered)} / {len(all_signals)}** signals passed ML filter")

                if filtered:
                    ml_data = []
                    for s in filtered:
                        # Safely extract attributes with fallback to None
                        symbol = getattr(s, 'symbol', None)
                        side = getattr(s, 'side', None)
                        score = getattr(s, 'score', None)
                        entry = getattr(s, 'entry', None)
                        ml_score = getattr(s, 'ml_score', None)  # or ml_probability, depending on your model

                        ml_data.append({
                            "Symbol": symbol or "N/A",
                            "Side": "LONG" if side == "LONG" else "SHORT",
                            "Score": f"{score:.1f}%" if score is not None else "N/A",
                            "Entry": f"${entry:,.4f}" if entry is not None else "N/A",
                            "ML Prob": f"{ml_score:.1%}" if ml_score is not None else "N/A"
                        })
                    
                    df = pd.DataFrame(ml_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No signals passed the ML confidence threshold.")

        except Exception as e:
            st.error(f"ML filter error: {e}")
            st.exception(e)  # optional: shows full traceback in Streamlit (great for debugging)

if __name__ == "__main__":
    if "engine" not in st.session_state:
        try:
            st.session_state.engine = TradingEngine()
        except Exception as e:
            st.error("Failed to initialize engine")
            st.stop()
    main()