# signal_generator.py
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
from indicators import scan_multiple_symbols, get_top_symbols, analyze_symbol_multi_tf
from ml import MLFilter
from notifications import send_all_notifications

from db import Signal, db_manager

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# Timezone for pretty timestamps (UTC+3)
tz_utc3 = timezone(timedelta(hours=3))

# ML Filter toggle
ML_ENABLED = True

# -------------------------------
# Core Signal Utilities (ALL PRESERVED)
# -------------------------------

def get_usdt_symbols(limit: int = 50) -> List[str]:
    """Get top USDT perpetuals – fallback if API fails"""
    try:
        symbols = get_top_symbols(limit)
        if not symbols:
            logger.warning("No symbols from API, using fallback list")
            symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        return symbols[:limit]
    except Exception as e:
        logger.error(f"Error fetching USDT symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
    """
    Kept for backward compatibility & ML filter.
    Now just returns the score already calculated in indicators.py
    """
    return float(analysis.get("score", 0))

def enhance_signal(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    NO LONGER NEEDED — multi-TF already returns perfect fields.
    But kept for compatibility — just returns the dict unchanged.
    """
    return analysis

# -------------------------------
# Signal Summary (PRESERVED)
# -------------------------------

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not signals:
        return {"total": 0, "avg_score": 0, "top_symbol": "None"}

    total_signals = len(signals)
    avg_score = sum(float(s.get("score", 0)) for s in signals) / total_signals
    top_signal = max(signals, key=lambda x: float(x.get("score", 0)))
    top_symbol = top_signal.get("symbol", "Unknown")

    buy_signals = sum(1 for s in signals if s.get("side", "").upper() == "LONG")
    sell_signals = total_signals - buy_signals

    market_types = {}
    for s in signals:
        market_types[s.get("market", "Unknown")] = market_types.get(s.get("market", "Unknown"), 0) + 1

    return {
        "total": total_signals,
        "avg_score": round(avg_score, 1),
        "top_symbol": top_symbol,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "market_distribution": market_types
    }

# -------------------------------
# Single Symbol Analysis (PRESERVED)
# -------------------------------

def analyze_single_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    """
    Analyze a single symbol using the new multi-TF logic
    """
    result = analyze_symbol_multi_tf(symbol)
    if not result:
        logger.warning(f"No multi-TF signal for {symbol}")
        return {}

    # Save to DB
    try:
        signal_obj = Signal(
            symbol=result["symbol"],
            interval="60",
            signal_type=result["type"],
            score=result["score"],
            indicators=result.get("indicators", {}),
            side=result["side"],
            sl=result["sl"],
            tp=result["tp"],
            trail=result["trail"],
            liquidation=result["liquidation"],
            leverage=20,
            margin_usdt=result["margin_usdt"],
            entry=result["entry"],
            market=result["market"],
            created_at=datetime.now(timezone.utc)
        )
        db_manager.add_signal(signal_obj)
    except Exception as e:
        logger.error(f"Failed to save {symbol} to DB: {e}")

    return result

# -------------------------------
# NEW: Beautiful Signal Formatting (Exact same as your working script)
# -------------------------------

def format_signal_block(s: Dict[str, Any]) -> str:
    side_emoji = "BUY" if s['side'] == "LONG" else "SELL"
    return (
        f"==================== {s['symbol']} ====================\n"
        f"TYPE: {s['type']}     SIDE: {side_emoji}     SCORE: {s['score']}%\n"
        f"ENTRY: {s['entry']}   TP: {s['tp']}         SL: {s['sl']}\n"
        f"MARKET: {s['market']}  BB: {s['bb_slope']}    Trail: {s['trail']}\n"
        f"MARGIN: {s['margin_usdt']}  LIQ: {s['liquidation']}    TIME: {s['time']}\n"
        "=========================================================\n"
    )

# -------------------------------
# Main Signal Generation (NOW USES YOUR WORKING LOGIC)
# -------------------------------

def generate_signals(
    symbols: List[str] | None = None,
    interval: str = "60",
    top_n: int = 10,
    trading_mode: str = "virtual"
) -> List[Dict[str, Any]]:
    """
    Generates signals exactly like your reference script:
    - Multi-timeframe alignment (15m, 60m, 240m)
    - Perfect LONG & SHORT
    - Clean formatting
    """
    logger.info(f"Starting signal generation in {trading_mode.upper()} mode")

    if not symbols:
        symbols = get_usdt_symbols(limit=100)

    logger.info(f"Scanning {len(symbols)} symbols using multi-timeframe alignment...")

    # This now uses the new analyze_symbol_multi_tf() from indicators.py
    raw_signals = scan_multiple_symbols(symbols, max_workers=8)

    if not raw_signals:
        logger.info("No multi-timeframe aligned signals found")
        return []

    # Optional: ML filtering (still works)
    if ML_ENABLED:
        try:
            ml_filter = MLFilter()
            raw_signals = ml_filter.filter_signals(raw_signals)
            logger.info(f"ML filter applied: {len(raw_signals)} signals remain")
        except Exception as e:
            logger.warning(f"ML filter failed: {e}")

    # Sort by score and take top N
    raw_signals.sort(key=lambda x: x.get("score", 0), reverse=True)
    final_signals = raw_signals[:top_n]

    # Save all to DB
    for sig in final_signals:
        try:
            db_signal = Signal(
                symbol=sig["symbol"],
                interval="60",
                signal_type=sig["type"],
                score=sig["score"],
                indicators=sig.get("indicators", {}),
                side=sig["side"],
                entry=sig["entry"],
                sl=sig["sl"],
                tp=sig["tp"],
                trail=sig["trail"],
                liquidation=sig["liquidation"],
                leverage=20,
                margin_usdt=sig["margin_usdt"],
                market=sig["market"],
                created_at=datetime.now(timezone.utc)
            )
            db_manager.add_signal(db_signal)
        except Exception as e:
            logger.error(f"DB save failed for {sig['symbol']}: {e}")

    # Print beautiful blocks
    if final_signals:
        blocks = [format_signal_block(s) for s in final_signals]
        message = "\n".join(blocks)
        logger.info(f"\n{message}")
        send_all_notifications(final_signals)

    # Summary
    summary = get_signal_summary(final_signals)
    logger.info(f"Signal Summary: {summary}")

    return final_signals


# -------------------------------
# Run Standalone (PRESERVED + Improved Logging)
# -------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Bybit Multi-Timeframe Signal Scanner")
    logger.info("Using 15m / 60m / 240m alignment – same as your working script")

    symbols = get_usdt_symbols(limit=100)
    signals = generate_signals(symbols=symbols, top_n=5, trading_mode="virtual")

    if signals:
        logger.info(f"Found {len(signals)} high-quality signals!")
    else:
        logger.info("No signals this hour – waiting for alignment")