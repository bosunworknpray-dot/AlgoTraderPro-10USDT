from datetime import datetime, timedelta, timezone

from engine import TradingEngine


def make_engine():
    engine = TradingEngine.__new__(TradingEngine)
    engine.settings = {
        "MIN_SIGNAL_SCORE": 50,
        "COOLDOWN_MINUTES": 30,
        "MAX_POSITION_SIZE": 10000,
        "MAX_OPEN_POSITIONS": 10,
        "MAX_RISK_PER_TRADE": 0.05,
    }
    engine.max_position_size = 10000
    engine.max_open_positions = 10
    engine.max_risk_per_trade = 0.05
    engine._trade_cooldowns = {}
    engine._consecutive_failures = 0
    engine._daily_pnl = 0.0
    return engine


def test_blocks_low_score_and_cooldown_signals():
    engine = make_engine()

    low_score_signal = {"symbol": "BTCUSDT", "score": 40}
    allowed, reason = engine.evaluate_trade_signal(low_score_signal, trading_mode="virtual", open_trades=[])
    assert allowed is False
    assert "score" in reason.lower()

    recent_signal = {"symbol": "ETHUSDT", "score": 80}
    engine._trade_cooldowns["ETHUSDT"] = datetime.now(timezone.utc) - timedelta(minutes=10)
    allowed, reason = engine.evaluate_trade_signal(recent_signal, trading_mode="virtual", open_trades=[])
    assert allowed is False
    assert "cooldown" in reason.lower()
