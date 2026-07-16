from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# === CONFIGURATION - Scalping defaults ===
INTERVALS = ['5', '15']
MAX_SYMBOLS = 100
MIN_VOLUME = 1000
MIN_ATR_PCT = 0.001
RSI_ZONE = (20, 80)
ACCOUNT_BALANCE = 100
RISK_PCT = 0.10
LEVERAGE = 10
ENTRY_BUFFER_PCT = 0.002

tz_utc3 = timezone(timedelta(hours=3))

# -------------------------------
# ORIGINAL FULLY PRESERVED INDICATOR FUNCTIONS
# -------------------------------

def get_candles(symbol: str, interval: str, limit: int = 200) -> List[Dict]:
    """Fetch candlestick data from Bybit API with retry logic"""
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }
        
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") == 0 and "result" in data:
            klines = []
            for k in data["result"]["list"]:
                klines.append({
                    "time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
            klines = sorted(klines, key=lambda x: x["time"])  # oldest first
            return klines
        else:
            logger.warning(f"Bybit API error for {symbol}: {data.get('retMsg', 'Unknown')}")
            return []
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol} {interval}: {e}")
        return []

def sma(prices: List[float], period: int) -> List[float]:
    """Simple Moving Average - full array"""
    if len(prices) < period:
        return [0.0] * len(prices)
    sma_values = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_values.append(0.0)
        else:
            avg = sum(prices[i-period+1:i+1]) / period
            sma_values.append(avg)
    return sma_values

def ema(prices: List[float], period: int) -> List[float]:
    """Exponential Moving Average - full array"""
    if len(prices) < period:
        return [0.0] * len(prices)
    multiplier = 2 / (period + 1)
    ema_values = [prices[0]]
    for i in range(1, len(prices)):
        val = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(val)
    return ema_values

def rsi(prices: List[float], period: int = 14) -> List[float]:
    """Relative Strength Index - full array"""
    if len(prices) < period + 1:
        return [50.0] * len(prices)
    
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(change if change > 0 else 0.0)
        losses.append(-change if change < 0 else 0.0)
    
    rsi_values = [50.0]
    for i in range(period, len(gains)):
        avg_gain = sum(gains[i-period:i]) / period
        avg_loss = sum(losses[i-period:i]) / period
        if avg_loss == 0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        rsi_values.append(rsi_val)
    
    while len(rsi_values) < len(prices):
        rsi_values.insert(0, 50.0)
    return rsi_values

def stochastic_rsi(prices: List[float], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, List[float]]:
    """Stochastic RSI - full arrays"""
    if len(prices) < period + 1:
        z = [50.0] * len(prices)
        return {"stoch_rsi": z, "k": z, "d": z}
    
    rsi_vals = rsi(prices, period)
    stoch = []
    for i in range(period-1, len(rsi_vals)):
        window = rsi_vals[i-period+1:i+1]
        low_rsi = min(window)
        high_rsi = max(window)
        current = rsi_vals[i]
        if high_rsi == low_rsi:
            stoch.append(50.0)
        else:
            stoch.append(100 * (current - low_rsi) / (high_rsi - low_rsi))
    
    while len(stoch) < len(prices):
        stoch.insert(0, 50.0)
    
    k = sma(stoch, smooth_k)
    d = sma(k, smooth_d)
    return {"stoch_rsi": stoch, "k": k, "d": d}

def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
    """Bollinger Bands - full arrays"""
    if len(prices) < period:
        z = [0.0] * len(prices)
        return {"upper": z, "middle": z, "lower": z}
    
    sma_vals = sma(prices, period)
    upper = []
    lower = []
    for i in range(len(prices)):
        if i < period - 1:
            upper.append(0.0)
            lower.append(0.0)
        else:
            segment = prices[i-period+1:i+1]
            std = np.std(segment)
            mid = sma_vals[i]
            upper.append(mid + std_dev * std)
            lower.append(mid - std_dev * std)
    
    return {"upper": upper, "middle": sma_vals, "lower": lower}

def calculate_indicators(candles: List[Dict]) -> Dict[str, Any]:
    """Full original indicator suite - preserved"""
    try:
        if not candles or len(candles) < 20:
            logger.warning(f"Insufficient candles: {len(candles)}")
            return {}
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        sma_20 = sma(closes, 20)
        sma_200 = sma(closes, 200) if len(closes) >= 200 else sma(closes, len(closes))
        ema_9 = ema(closes, 9)
        ema_21 = ema(closes, 21)
        rsi_14 = rsi(closes, 14)
        stoch = stochastic_rsi(closes)
        bb = bollinger_bands(closes)
        
        current_price = closes[-1]
        avg_vol = sum(volumes[-20:]) / min(20, len(volumes))
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        
        trend_score = 0
        if len(sma_20) > 1:
            if sma_20[-1] > sma_200[-1]: trend_score += 1
            if closes[-1] > sma_20[-1]: trend_score += 1
            if sma_20[-1] > sma_20[-2]: trend_score += 1
            if ema_9[-1] > ema_21[-1]: trend_score += 1
        
        bb_width = (bb["upper"][-1] - bb["lower"][-1]) / current_price * 100 if current_price > 0 else 0
        
        return {
            "price": current_price,
            "sma_20": sma_20[-1],
            "sma_200": sma_200[-1],
            "ema_9": ema_9[-1],
            "ema_21": ema_21[-1],
            "rsi": rsi_14[-1],
            "stoch_k": stoch["k"][-1],
            "stoch_d": stoch["d"][-1],
            "bb_upper": bb["upper"][-1],
            "bb_lower": bb["lower"][-1],
            "bb_middle": bb["middle"][-1],
            "volume": volumes[-1],
            "volume_ratio": vol_ratio,
            "trend_score": trend_score,
            "volatility": bb_width
        }
    except Exception as e:
        logger.error(f"Error in calculate_indicators: {e}")
        return {}

def get_top_symbols(limit: int = 100) -> List[str]:
    """Preserved original top symbols function"""
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear"}
        resp = session.get(url, params=params, timeout=15).json()
        
        if resp.get("retCode") != 0:
            return ["BTCUSDT", "ETHUSDT"]
        
        tickers = [t for t in resp["result"]["list"] if t["symbol"].endswith("USDT")]
        tickers.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
        return [t["symbol"] for t in tickers[:limit]]
    except Exception as e:
        logger.error(f"Error get_top_symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# -------------------------------
# NEW: Multi-TF Signal Logic (Your Working Script Style)
# -------------------------------

def ema_single(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period: return None
    k = 2 / (period + 1)
    val = prices[0]
    for p in prices[1:]:
        val = p * k + val * (1 - k)
    return val

def sma_single(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period: return None
    return sum(prices[-period:]) / period

def rsi_single(prices: List[float], period=14) -> Optional[float]:
    if len(prices) < period + 1: return None
    gains = losses = 0
    for i in range(1, period+1):
        diff = prices[-i] - prices[-i-1]
        if diff > 0: gains += diff
        else: losses -= diff
    avg_g = gains / period
    avg_l = losses / period or 1e-10
    return 100 - (100 / (1 + avg_g/avg_l))

def bb_single(prices: List[float], period=20, std=2) -> tuple:
    mid = sma_single(prices, period)
    if not mid: return None, None, None
    std_val = np.std(prices[-period:])
    return mid + std*std_val, mid, mid - std*std_val

def atr_single(highs, lows, closes, period=14) -> Optional[float]:
    if len(highs) < period + 1: return None
    trs = [max(h-l, abs(h-c), abs(l-c)) for h,l,c in zip(highs[1:], lows[1:], closes[:-1])]
    return sum(trs[-period:]) / period

def macd_single(prices: List[float]) -> Optional[float]:
    e12 = ema_single(prices, 12)
    e26 = ema_single(prices, 26)
    return (e12 - e26) if e12 and e26 else None

def classify_trend(ema9, ema21, sma20) -> str:
    if ema9 > ema21 > sma20: return "Trend"
    if ema9 > ema21: return "Swing"
    return "Scalp"

def analyze_symbol_multi_tf(symbol: str) -> Optional[Dict[str, Any]]:
    """Exact replica of your working script - multi-TF aligned signals"""
    try:
        data = {}
        for tf in INTERVALS:
            candles = get_candles(symbol, tf, 200)
            if len(candles) < 50: return None
            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]
            volumes = [c["volume"] for c in candles]
            
            bb_u, bb_m, bb_l = bb_single(closes)
            data[tf] = {
                "close": closes[-1],
                "ema9": ema_single(closes, 9),
                "ema21": ema_single(closes, 21),
                "sma20": sma_single(closes, 20),
                "rsi": rsi_single(closes),
                "macd": macd_single(closes),
                "bb_u": bb_u,
                "bb_m": bb_m,
                "bb_l": bb_l,
                "atr": atr_single(highs, lows, closes),
                "volume": volumes[-1]
            }

        p60 = data['60']
        if (p60['volume'] < MIN_VOLUME or
            p60['atr']/p60['close'] < MIN_ATR_PCT or
            not (RSI_ZONE[0] < p60['rsi'] < RSI_ZONE[1])):
            return None

        directions = []
        for tf in INTERVALS:
            d = data[tf]
            if d['close'] > d['bb_u']: directions.append("LONG")
            elif d['close'] < d['bb_l']: directions.append("SHORT")
            elif d['close'] > d['ema21']: directions.append("LONG")
            elif d['close'] < d['ema21']: directions.append("SHORT")

        if len(set(directions)) != 1: return None

        side = directions[0]
        price = p60['close']
        cands = [v for v in [p60['sma20'], p60['ema9'], p60['ema21']] if v]
        entry = min(cands, key=lambda x: abs(x - price))
        bb_slope = "Up" if price > p60['bb_u'] else "Down" if price < p60['bb_l'] else "Neutral"

        tp = round(entry * (1.10 if side=="LONG" else 0.90), 6)
        sl = round(entry * (0.95 if side=="LONG" else 1.05), 6)
        trail = round(entry * (1-ENTRY_BUFFER_PCT if side=="LONG" else 1+ENTRY_BUFFER_PCT), 6)
        liq = round(entry * (1 - 1/LEVERAGE if side=="LONG" else 1 + 1/LEVERAGE), 6)

        try:
            sl_diff = abs(entry - sl)
            margin = round((ACCOUNT_BALANCE * RISK_PCT / sl_diff) * entry / LEVERAGE, 6)
        except: margin = 10.0

        score = 0.3*(p60['macd']>0) + 0.2*(p60['rsi']<30 or p60['rsi']>70) + \
                 0.3*(bb_slope!="Neutral") + 0.2*(classify_trend(p60['ema9'], p60['ema21'], p60['sma20'])=="Trend") + 0.1

        return {
            "symbol": symbol,
            "side": side,
            "type": classify_trend(p60['ema9'], p60['ema21'], p60['sma20']),
            "score": round(score*100, 1),
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "trail": trail,
            "margin_usdt": margin,
            "liquidation": liq,
            "market": f"{price:.6f}",
            "bb_slope": bb_slope,
            "time": datetime.now(tz_utc3).strftime("%Y-%m-%d %H:%M UTC+3"),
            "indicators": data
        }
    except Exception as e:
        logger.error(f"Multi-TF failed {symbol}: {e}")
        return None

# -------------------------------
# Keep original functions for backward compatibility
# -------------------------------

def analyze_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    result = analyze_symbol_multi_tf(symbol)
    if result: return result
    # fallback to old logic
    candles = get_candles(symbol, interval, 200)
    ind = calculate_indicators(candles)
    return {"symbol": symbol, "score": 0, "signal_type": "neutral", "side": "Buy", "indicators": ind}

def scan_multiple_symbols(symbols: List[str], interval: str = "60", max_workers: int = 10) -> List[Dict]:
    """Now uses multi-TF logic"""
    logger.info(f"Scanning {len(symbols)} symbols with multi-timeframe alignment...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_symbol_multi_tf, s): s for s in symbols}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res: results.append(res)
            except: pass
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Multi-TF scan complete: {len(results)} signals")
    return results