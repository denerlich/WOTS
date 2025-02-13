import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import datetime
import plotly.graph_objects as go
import io
import yfinance as yf
import asyncio
import nest_asyncio
import logging

# === Consolidated Logging Configuration ===
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === Single nest_asyncio Application ===
nest_asyncio.apply()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import *
pd.options.display.float_format = '{:.2f}'.format

# ========== IBKR CONNECTION & DATA FETCHING ==========

def connect_ibkr():
    """Connect to IBKR via ib_insync with a session-based clientId."""
    ib = IB()
    
    # Generate a random client ID if not already generated
    if "ibkrClientId" not in st.session_state:
        st.session_state["ibkrClientId"] = random.randint(1000, 9999)
    
    ib.connect("127.0.0.1", 7496, clientId=st.session_state["ibkrClientId"])
    logger.debug("Connected to IBKR")
    return ib

def fetch_ibkr_stock_data(ib, ticker, duration='1 Y'):
    """
    Fetch historical daily bars from IBKR for `ticker`.
    Returns a dataframe with columns: Open, High, Low, Close, Volume, indexed by date.
    """
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True
    )
    if not bars:
        return None
    df = util.df(bars)
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    df.set_index('date', inplace=True)
    logger.debug("Fetched IBKR historical data with %d rows", len(df))
    return df

def fetch_ibkr_implied_volatility(ib, ticker):
    """
    Get a snapshot of implied volatility from IBKR’s live market data.
    If not available, returns None.
    """
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    ticker_data = ib.reqMktData(contract, '', False, False)
    ib.sleep(2)
    if ticker_data.modelGreeks and ticker_data.modelGreeks.impliedVolatility is not None:
        logger.debug("Fetched IBKR implied volatility: %s", ticker_data.modelGreeks.impliedVolatility)
        return ticker_data.modelGreeks.impliedVolatility
    logger.debug("IBKR implied volatility not available")
    return None

def fetch_ibkr_option_chain(ib, ticker):
    """
    Fetch option chain parameters from IBKR.
    Returns the raw data.
    """
    contract = Option(ticker, '', 0, '', 'SMART')
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=ticker,
        futFopExchange='',
        underlyingSecType='STK',
        underlyingConId=0
    )
    logger.debug("Fetched IBKR option chain data: %s", chains)
    return chains

# ========== FINANCIAL MODELING PREP & ALPHA VANTAGE FALLBACKS ==========

FMP_API_KEY = "XTP3SR4gSz4vigQTIQsYesnBkGbVySVi"
ALPHA_VANTAGE_API_KEY = "YRHEPCO88AQSJFVG"

def is_missing(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        logging.warning(f"Missing value detected: {value}")
        return True
    return False

def fetch_fmp_ratios(ticker):
    """Fetch fundamental ratios from Financial Modeling Prep."""
    url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={FMP_API_KEY}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.debug("FMP ratios raw data: %s", data)
            if data:
                latest = data[0]
                return {
                    'Debt-to-Equity Ratio': latest.get('debtEquityRatio'),
                    'Current Ratio': latest.get('currentRatio'),
                    'Return on Equity (%)': latest.get('returnOnEquity') * 100 if latest.get('returnOnEquity') else None,
                    'PEG Ratio': latest.get('pegRatio')
                }
    except requests.exceptions.RequestException as e:
        logger.error("FMP API error: %s", e)
    return {}

def fetch_fmp_key_metrics(ticker):
    """Fetch key financial metrics from Financial Modeling Prep."""
    url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={FMP_API_KEY}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.debug("FMP key metrics raw data: %s", data)
            if data:
                latest = data[0]
                return {'Free Cash Flow': latest.get('freeCashFlowPerShare')}
    except requests.exceptions.RequestException as e:
        logger.error("FMP API error: %s", e)
    return {}

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def fetch_alpha_vantage_overview(ticker):
    """Fetch company overview from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.debug("Alpha Vantage raw data: %s", data)
            return {
                'Debt-to-Equity Ratio': safe_float(data.get('DebtToEquity')),
                'Current Ratio': safe_float(data.get('CurrentRatio')),
                'Return on Equity (%)': safe_float(data.get('ReturnOnEquityTTM')),
                'PEG Ratio': safe_float(data.get('PEGRatio'))
            }
    except requests.exceptions.RequestException as e:
        logger.error("Alpha Vantage API error: %s", e)
    return {}

def get_fundamental_data(ticker):
    """
    Retrieve fundamental data from Yahoo Finance.
    For any missing field, use fallback sources with the following priorities:
      1. For Debt-to-Equity Ratio, Current Ratio, and ROE: use FMP ratios then Alpha Vantage.
      2. For Free Cash Flow: use FMP key metrics.
    """
    stock = yf.Ticker(ticker)
    yahoo_info = stock.info
    logger.debug("Yahoo Finance info: %s", yahoo_info)
    fundamentals = {
        "Debt-to-Equity Ratio": yahoo_info.get("debtToEquity", np.nan),
        "Current Ratio": yahoo_info.get("currentRatio", np.nan),
        "Free Cash Flow": yahoo_info.get("freeCashFlow", np.nan),
        "Return on Equity (%)": yahoo_info.get("returnOnEquity", np.nan),
        "PEG Ratio": yahoo_info.get("pegRatio", np.nan),
    }

    # Fallback using FMP ratios
    fmp_ratios = fetch_fmp_ratios(ticker)
    logger.debug("Using FMP ratios fallback: %s", fmp_ratios)

    for key in ["Debt-to-Equity Ratio", "Current Ratio", "Return on Equity (%)"]:
        if is_missing(fundamentals[key]) and not is_missing(fmp_ratios.get(key)):
            fundamentals[key] = fmp_ratios[key]
            logger.info(f"Filled {key} from FMP: {fundamentals[key]}")

    # Fallback for Free Cash Flow using FMP key metrics
    fmp_key_metrics = fetch_fmp_key_metrics(ticker)
    logger.debug("Using FMP key metrics fallback: %s", fmp_key_metrics)

    if is_missing(fundamentals["Free Cash Flow"]):
        if not is_missing(fmp_key_metrics.get("Free Cash Flow")):
            fundamentals["Free Cash Flow"] = fmp_key_metrics["Free Cash Flow"]
            logger.info(f"Filled Free Cash Flow from FMP key metrics: {fundamentals['Free Cash Flow']}")
        else:
            fundamentals["Free Cash Flow"] = np.nan  # Explicitly setting NaN if missing from all sources

    # Fallback using Alpha Vantage for remaining fields
    alpha_ratios = fetch_alpha_vantage_overview(ticker)
    logger.debug("Using Alpha Vantage fallback: %s", alpha_ratios)
    if is_missing(fundamentals["Debt-to-Equity Ratio"]) and not is_missing(alpha_ratios.get("Debt-to-Equity Ratio")):
        fundamentals["Debt-to-Equity Ratio"] = alpha_ratios["Debt-to-Equity Ratio"]
        logger.debug("Filled Debt-to-Equity Ratio from Alpha Vantage: %s", fundamentals["Debt-to-Equity Ratio"])
    if is_missing(fundamentals["Current Ratio"]) and not is_missing(alpha_ratios.get("Current Ratio")):
        fundamentals["Current Ratio"] = alpha_ratios["Current Ratio"]
        logger.debug("Filled Current Ratio from Alpha Vantage: %s", fundamentals["Current Ratio"])
    if is_missing(fundamentals["Return on Equity (%)"]) and not is_missing(alpha_ratios.get("Return on Equity (%)")):
        fundamentals["Return on Equity (%)"] = alpha_ratios["Return on Equity (%)"]
        logger.debug("Filled Return on Equity (%) from Alpha Vantage: %s", fundamentals["Return on Equity (%)"])
    if is_missing(fundamentals["PEG Ratio"]) and not is_missing(alpha_ratios.get("PEG Ratio")):
        fundamentals["PEG Ratio"] = alpha_ratios["PEG Ratio"]
        logger.debug("Filled PEG Ratio from Alpha Vantage: %s", fundamentals["PEG Ratio"])

    logger.info("Final fundamental data after fallbacks: %s", fundamentals)

    # Final cleanup: replace any remaining NaN values with "N/A"
    for key in fundamentals:
        if is_missing(fundamentals[key]):
            fundamentals[key] = "N/A"

    logger.info("Final cleaned fundamental data: %s", fundamentals)
    return fundamentals

# ========== ORIGINAL FUNDAMENTAL & TECHNICAL CALCULATIONS ==========

def calculate_fundamentals(ticker_obj):
    """Compute advanced fundamentals using yfinance's balance_sheet, cashflow, and financials."""
    try:
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        fs = ticker_obj.financials
    except Exception as e:
        st.error(f"Error retrieving fundamental data: {e}")
        logger.error("Error retrieving fundamental data: %s", e)
        return {}
    
    try:
        latest_bs = bs.iloc[:, 0]
    except Exception:
        latest_bs = pd.Series()

    total_liabilities = latest_bs.get('Total Liab', np.nan)
    shareholder_equity = latest_bs.get('Total Stockholder Equity', np.nan)
    debt_to_equity = (
        total_liabilities / shareholder_equity
        if pd.notna(total_liabilities) and pd.notna(shareholder_equity) and shareholder_equity != 0
        else np.nan
    )

    current_assets = latest_bs.get('Total Current Assets', np.nan)
    current_liabilities = latest_bs.get('Total Current Liabilities', np.nan)
    current_ratio = (
        current_assets / current_liabilities
        if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0
        else np.nan
    )

    try:
        latest_cf = cf.iloc[:, 0]
        operating_cf = latest_cf.get("Total Cash From Operating Activities", np.nan)
        capex = latest_cf.get("Capital Expenditures", np.nan)
        free_cash_flow = (
            operating_cf - capex
            if pd.notna(operating_cf) and pd.notna(capex)
            else np.nan
        )
    except Exception:
        free_cash_flow = np.nan

    revenue_growth = np.nan
    if fs.shape[1] >= 2:
        rev_rows = [row for row in fs.index if "Total Revenue" in row]
        if rev_rows:
            latest_revenue = fs.loc[rev_rows[0], fs.columns[0]]
            previous_revenue = fs.loc[rev_rows[0], fs.columns[1]]
            if pd.notna(latest_revenue) and pd.notna(previous_revenue) and previous_revenue != 0:
                revenue_growth = ((latest_revenue - previous_revenue) / previous_revenue) * 100

    net_income = np.nan
    total_revenue = np.nan
    for row in fs.index:
        if "Net Income" in row:
            net_income = fs.loc[row, fs.columns[0]]
        if "Total Revenue" in row:
            total_revenue = fs.loc[row, fs.columns[0]]
    profit_margin = (
        (net_income / total_revenue) * 100
        if pd.notna(net_income) and pd.notna(total_revenue) and total_revenue != 0
        else np.nan
    )
    roe = (
        (net_income / shareholder_equity) * 100
        if pd.notna(net_income) and pd.notna(shareholder_equity) and shareholder_equity != 0
        else np.nan
    )

    fundamentals = {
        "Debt-to-Equity Ratio": debt_to_equity,
        "Current Ratio": current_ratio,
        "Free Cash Flow": free_cash_flow,
        "Revenue Growth (%)": revenue_growth,
        "Profit Margin (%)": profit_margin,
        "Return on Equity (%)": roe,
    }
    logger.debug("Calculated advanced fundamentals: %s", fundamentals)
    return fundamentals

def get_extra_fundamentals(ticker_obj):
    """Pull additional metrics from Yahoo Finance's info dict."""
    info = ticker_obj.info
    extra = {
        "Beta": info.get("beta", np.nan),
        "Trailing P/E": info.get("trailingPE", np.nan),
        "Forward P/E": info.get("forwardPE", np.nan),
        "PEG Ratio": info.get("pegRatio", np.nan),
        "Dividend Yield": info.get("dividendYield", np.nan),
        "Market Cap": info.get("marketCap", np.nan),
        "52-Week Change": info.get("52WeekChange", np.nan)
    }
    logger.debug("Extra fundamentals: %s", extra)
    return extra

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return sma.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]

def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv[-1]

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return adx.iloc[-1]

def calculate_roc(series, period=12):
    if len(series) >= period:
        return ((series.iloc[-1] / series.iloc[-period]) - 1) * 100
    return np.nan

def calculate_technical_indicators(hist):
    """
    Compute technical indicators using historical data.
    """
    if hist.empty:
        st.error("No historical data available.")
        return {}
    
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()

    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    hist['MACD'] = macd_line
    hist['MACD_Signal'] = signal_line
    hist['MACD_Hist'] = macd_hist
    
    rsi = calculate_rsi(hist['Close'])
    ma_sma, upper_bb, lower_bb = calculate_bollinger_bands(hist['Close'])
    obv_value = calculate_obv(hist)
    adx_value = calculate_adx(hist)
    roc_value = calculate_roc(hist['Close'])
    
    latest = hist.iloc[-1]
    technicals = {
        "Current Price": latest['Close'],
        "MA20": latest['MA20'],
        "MA50": latest['MA50'],
        "MA200": latest['MA200'],
        "MACD": latest['MACD'],
        "MACD Signal": latest['MACD_Signal'],
        "MACD Histogram": latest['MACD_Hist'],
        "RSI": rsi,
        "Bollinger Middle Band": ma_sma,
        "Bollinger Upper Band": upper_bb,
        "Bollinger Lower Band": lower_bb,
        "OBV": obv_value,
        "ADX": adx_value,
        "Rate of Change (%)": roc_value
    }
    logger.debug("Calculated technical indicators: %s", technicals)
    return technicals

def calculate_options_indicators_ibkr(ib, ticker):
    """
    Retrieve options data from IBKR. For simplicity, we return a single snapshot IV.
    """
    iv_snapshot = fetch_ibkr_implied_volatility(ib, ticker)
    logger.debug("Options data (IV): %s", iv_snapshot)
    return {
        "Options Expiration": "N/A (custom logic needed)",
        "Average IV (Puts)": iv_snapshot,
        "Average IV (Calls)": iv_snapshot
    }

# ========== UI HELPER FUNCTIONS & FINAL VERDICT ==========

def get_indicator_color(value, favorable, borderline, higher_is_better=True):
    try:
        if np.isnan(value):
            return "gray"
        if higher_is_better:
            if value >= favorable:
                return "green"
            elif value >= borderline:
                return "yellow"
            else:
                return "red"
        else:
            if value <= favorable:
                return "green"
            elif value <= borderline:
                return "yellow"
            else:
                return "red"
    except Exception:
        return "gray"

def get_icon(detail):
    lower = detail.lower()
    if any(word in lower for word in ["ok", "positive", "strong", "good", "optimal", "favorable"]):
        return "👍"
    elif any(word in lower for word in ["high", "weak", "negative", "low"]):
        return "👎"
    else:
        return "🤷"

def final_verdict(fund, tech, extra, opts, thresholds):
    verdict = []
    if fund.get("Debt-to-Equity Ratio", np.nan) <= thresholds["Debt-to-Equity"]:
        verdict.append("Debt-to-Equity OK")
    else:
        verdict.append("High Debt-to-Equity")
    if fund.get("Current Ratio", 0) >= thresholds["Current Ratio"]:
        verdict.append("Current Ratio OK")
    else:
        verdict.append("Weak Liquidity")
    if fund.get("Free Cash Flow", 0) > 0:
        verdict.append("Positive Free Cash Flow")
    else:
        verdict.append("Negative Free Cash Flow")
    if fund.get("Revenue Growth (%)", 0) >= thresholds["Revenue Growth"]:
        verdict.append("Strong Revenue Growth")
    else:
        verdict.append("Weak Revenue Growth")
    if fund.get("Return on Equity (%)", 0) >= thresholds["ROE"]:
        verdict.append("Good ROE")
    else:
        verdict.append("Low ROE")
    
    if tech.get("RSI", 0) >= thresholds["RSI_low"] and tech.get("RSI", 0) <= thresholds["RSI_high"]:
        verdict.append("RSI is Optimal")
    else:
        verdict.append("RSI is Suboptimal")
    if tech.get("MACD Histogram", 0) > 0:
        verdict.append("MACD Indicates Bullish Momentum")
    else:
        verdict.append("MACD Indicates Bearish/Neutral Momentum")
    
    iv_value = opts.get("Average IV (Puts)")
    if iv_value is None:
        iv_value = np.inf
    if iv_value <= thresholds["IV"]:
        verdict.append("Options IV is Favorable")
    else:
        verdict.append("Options IV is High")
    
    favorable_count = sum(
        1 for v in verdict
        if any(x in v for x in ["OK", "Positive", "Strong", "Good", "Optimal", "Favorable"])
    )
    final_str = "Good Candidate for Selling Puts" if favorable_count >= 5 else "Not Suitable for Selling Puts"
    logger.debug("Final verdict: %s, details: %s", final_str, verdict)
    return final_str, verdict

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="Options Wheel Trade Checker", layout="wide")
st.title("Options Wheel Trade Stock Assessment (IBKR + Fallback)")

st.sidebar.header("Adjust Thresholds (Click on '?' for details)")
de_threshold = st.sidebar.number_input("Max Debt-to-Equity Ratio", min_value=0.0, value=0.5, step=0.1,
                                       help="Favorable if debt-to-equity ratio is below this value.")
cr_threshold = st.sidebar.number_input("Min Current Ratio", min_value=0.0, value=1.5, step=0.1,
                                       help="Favorable if current ratio is above this value.")
rev_growth_threshold = st.sidebar.number_input("Min Revenue Growth (%)", value=10.0, step=1.0,
                                               help="Favorable if YoY revenue growth is above this percent.")
roe_threshold = st.sidebar.number_input("Min Return on Equity (%)", value=15.0, step=1.0,
                                        help="Favorable if ROE is above this percent.")
rsi_low = st.sidebar.number_input("RSI Lower Bound", min_value=0, max_value=100, value=40, step=1,
                                  help="Favorable if RSI is above this value.")
rsi_high = st.sidebar.number_input("RSI Upper Bound", min_value=0, max_value=100, value=60, step=1,
                                   help="Favorable if RSI is below this value.")
iv_threshold = st.sidebar.number_input("Max Average IV for Puts", min_value=0.0, value=0.50, step=0.01,
                                       help="Favorable if average implied volatility for puts is below this value (e.g. 0.50 = 50%).")

thresholds = {
    "Debt-to-Equity": de_threshold,
    "Current Ratio": cr_threshold,
    "Revenue Growth": rev_growth_threshold,
    "ROE": roe_threshold,
    "RSI_low": rsi_low,
    "RSI_high": rsi_high,
    "IV": iv_threshold,
}

ticker_input = st.text_input("Enter Ticker Symbol", value="AAPL")
run_button = st.button("Run Analysis")

if run_button and ticker_input:
    ticker = ticker_input.upper()
    ib = connect_ibkr()
    
    with st.spinner("Fetching data from IBKR..."):
        ibkr_hist = fetch_ibkr_stock_data(ib, ticker, duration='1 Y')
    
    if ibkr_hist is not None and not ibkr_hist.empty:
        hist = ibkr_hist
        st.success("Successfully retrieved historical data from IBKR.")
    else:
        st.warning("IBKR data not available. Falling back to Yahoo Finance.")
        yfa = yf.Ticker(ticker)
        hist = yfa.history(period="1y")

    with st.spinner("Fetching fundamentals..."):
        fundamentals_fallback = get_fundamental_data(ticker)
        yfa = yf.Ticker(ticker)
        adv_fund = calculate_fundamentals(yfa)
        adv_fund_extra = get_extra_fundamentals(yfa)
    
    with st.spinner("Analyzing technicals..."):
        technicals = calculate_technical_indicators(hist)
    
    with st.spinner("Fetching options data from IBKR..."):
        options_data = calculate_options_indicators_ibkr(ib, ticker)
    
    # After all IBKR data calls complete, disconnect from IBKR
    ib.disconnect()

    # Combine fundamentals from fallback and advanced
    combined_fundamentals = {}
    for key in set(fundamentals_fallback.keys()).union(adv_fund.keys()):
        fallback_value = fundamentals_fallback.get(key, np.nan)
        adv_value = adv_fund.get(key, np.nan)
        combined_fundamentals[key] = adv_value if not is_missing(adv_value) else fallback_value

    final_str, detail_list = final_verdict(combined_fundamentals, technicals, adv_fund_extra, options_data, thresholds)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div style='font-size:20px; font-weight:bold;'>Fundamental Indicators</div>", unsafe_allow_html=True)
        fund_df = pd.DataFrame(list(combined_fundamentals.items()), columns=["Indicator", "Value"])
        st.markdown(fund_df.to_html(classes="table", index=False), unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='font-size:20px; font-weight:bold;'>Technical Indicators</div>", unsafe_allow_html=True)
        tech_df = pd.DataFrame(list(technicals.items()), columns=["Indicator", "Value"])
        st.markdown(tech_df.to_html(classes="table", index=False), unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='font-size:20px; font-weight:bold;'>Final Verdict</div>", unsafe_allow_html=True)
        final_icon = "👍" if final_str == "Good Candidate for Selling Puts" else "👎"
        st.markdown(f"<div style='font-size:22px;'>{final_str} {final_icon}</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div style='font-size:20px; font-weight:bold;'>Detail Assessment</div>", unsafe_allow_html=True)
        detail_html = "<ul style='font-size:18px;'>"
        for detail in detail_list:
            icon = get_icon(detail)
            detail_html += f"<li>{icon} {detail}</li>"
        detail_html += "</ul>"
        st.markdown(detail_html, unsafe_allow_html=True)
    
    # Plot candle chart if data is available
    if hist is not None and not hist.empty:
        sma20 = hist['Close'].rolling(window=20).mean()
        std20 = hist['Close'].rolling(window=20).std()
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        
        fig_candle = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Price")])
        fig_candle.add_trace(go.Scatter(x=hist.index, y=sma20, line=dict(color='blue', width=1), name='MA20'))
        fig_candle.add_trace(go.Scatter(x=hist.index, y=upper_bb, line=dict(color='orange', width=1), name='Upper BB'))
        fig_candle.add_trace(go.Scatter(x=hist.index, y=lower_bb, line=dict(color='orange', width=1), name='Lower BB'))
        fig_candle.update_layout(title=f"{ticker} Candlestick Chart with Bollinger Bands",
                                 xaxis_title="Date", yaxis_title="Price")
        
        st.plotly_chart(fig_candle, use_container_width=True)
        
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=hist.index, y=macd_line, line=dict(color='blue', width=1), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=hist.index, y=signal_line, line=dict(color='orange', width=1), name="Signal"))
        fig_macd.add_trace(go.Bar(x=hist.index, y=macd_hist, name="Histogram"))
        fig_macd.update_layout(title="MACD Chart", xaxis_title="Date", yaxis_title="Value")
        
        st.plotly_chart(fig_macd, use_container_width=True)

        buf = io.BytesIO()
        try:
            fig_candle.write_image(buf, format="png")
            buf.seek(0)
            st.download_button(label="Save Snapshot (Candlestick Chart as PNG)",
                               data=buf,
                               file_name=f"{ticker}_snapshot.png",
                               mime="image/png")
        except Exception as e:
            st.error(f"Error saving snapshot: {e}")
    else:
        st.error("No historical price data available to plot.")
