# Options Wheel Strategy Filtering Pipeline

This project aims to identify and filter stocks suitable for implementing the **Options Wheel Strategy**. By leveraging public APIs and advanced filtering techniques, the script ensures efficient identification of stable and predictable stocks for options trading.

---

## **Overview**

### **What is the Options Wheel Strategy?**
The Options Wheel Strategy involves selling options (typically cash-secured puts or covered calls) on high-quality stocks. This strategy generates income from option premiums while potentially acquiring or holding stocks with solid fundamentals. Stocks that perform well in this strategy generally:
- Have stable price movements.
- Provide decent dividend yields.
- Offer liquid options markets with good implied volatility.

### **Objective of the Script**
This script:
1. Fetches all available US stock tickers (from NYSE and NASDAQ).
2. Filters stocks to identify options with weekly expiration dates for enhanced liquidity.
3. Uses derived volatility metrics to approximate and filter suitable stocks based on thresholds.
4. Applies layered filtering to narrow down stocks suitable for the Options Wheel Strategy.
5. Outputs a list of final candidates with strong fundamentals, stable metrics, and attractive options.

---

## **Key Features**

1. **Automated Ticker Fetching**:
   - Uses Financial Modeling Prep (FMP) API to fetch stock tickers in bulk.

2. **Layered Filtering**:
   - Preliminary filters:
     - Exchange: NYSE/NASDAQ.
     - Price: Stocks priced between $0 and $200.
     - Type: Excludes non-stocks (e.g., ETFs).
     - Options with Weekly Expirations: Focuses only on stocks with weekly options.
   - Secondary filters (via Yahoo Finance):
     - Implied Volatility Rank (IVR): ≥ 35% for selling options.
     - Derived Volatility Indicators:
       - **Volatility Range**: Based on 52-week high/low vs. current price.
       - **Beta**: Between 0.5 and 1.5.
       - **Volume Trends**: Recent volume spikes using a ratio of 10-day average to overall average volume.
     - Technical Indicators:
       - **RSI**: Below 35 for oversold conditions.
       - **Stochastic**: Below 25.
       - **Bollinger Bands**: Price near the lower band to identify potential upward trends.
     - Trend Identification: Stocks on a general uptrend but currently in a pullback.

3. **Comprehensive Logging**:
   - Logs all API requests, responses, and filtering decisions in a structured file.
   - Provides live progress updates for real-time monitoring.

4. **API Usage Optimization**:
   - Caches bulk data locally to reduce redundant calls.
   - Implements time delays to respect rate limits for APIs.

---

## **System Requirements**

### **Environment**
- Python 3.8 or higher.

### **Dependencies**
Install the required Python libraries:
```bash
pip install pandas requests yfinance
```

### **APIs Used**
1. **Financial Modeling Prep (FMP)**:
   - Fetches bulk stock tickers and fundamentals.
   - Free API Key: `XTP3SR4gSz4vigQTIQsYesnBkGbVySVi`.

2. **Yahoo Finance (via yfinance)**:
   - Fetches additional stock fundamentals and technical indicators like RSI, Stochastic, and Bollinger Bands.

3. **IBKR API or Alternative for Options Data**:
   - Cross-checks weekly expirations and fetches additional implied volatility metrics.

---

## **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set API Keys**:
   - Replace the placeholders in the script with your actual API keys.

3. **Run the Script**:
   Execute the script to start the pipeline:
   ```bash
   python options_wheel_pipeline.py
   ```

4. **Log File Location**:
   - All logs are saved to `C:\Obsidian\VBAScripts\IBKR\logs\IBKR.log`.

---

## **Workflow**

### **Step 1: Fetch Bulk Data**
- Fetch all stocks listed on NYSE and NASDAQ.
- Filter out non-stocks (e.g., ETFs) and high-priced stocks (> $200).
- Retain only stocks with weekly options expiration dates to ensure high liquidity.

### **Step 2: Apply Secondary Filters**
For the remaining stocks, fetch additional metrics from Yahoo Finance:
- **Implied Volatility Rank (IVR)** ≥ 35%.
- **Derived Volatility Indicators**:
  - **Volatility Range**: `(52-Week High - 52-Week Low) / Current Price`.
  - **Beta**: Between 0.5 and 1.5.
  - **Volume Trend Ratio**: `(10-Day Average Volume / Overall Average Volume)` > 1 indicates recent spikes.
- **Technical Indicators**:
  - **RSI** < 35.
  - **Stochastic** < 25.
  - **Bollinger Bands**: Price near the lower band.
- Identify stocks with an **uptrend but currently in a pullback** using moving averages or trendlines.

### **Step 3: Output Final Results**
- Generate a filtered list of stocks suitable for the Options Wheel Strategy.
- Display results in the console and log details to the file.

---

## **Detailed Logging**

### **Log File**
All steps, API requests, and filtering decisions are logged in `IBKR.log`.

### **Sample Log Entry**
```plaintext
2025-01-25 12:00:00 - INFO - Fetching bulk stock data from FMP...
2025-01-25 12:00:03 - INFO - Fetched 3,000 stocks.
2025-01-25 12:00:05 - INFO - 250 tickers remain after pre-filtering.
2025-01-25 12:00:06 - INFO - Fetching fundamentals for AAPL (1/250)...
2025-01-25 12:00:07 - INFO - AAPL: Passed all filters. Added to final list.
```

---

## **Strategy Overview**

### **Selection Criteria for Stocks**
To maximize predictability and premium yield:
1. **Implied Volatility Rank (IVR)**:
   - Should be ≥ 35% for optimal premium income.

2. **Liquidity**:
   - Focus only on stocks with weekly options expiration dates.
   - High open interest and option volume ensure better execution and tighter spreads.

3. **Derived Volatility Indicators**:
   - **Volatility Range**: High range values indicate more movement.
   - **Volume Trend Ratio**: Indicates increasing activity and interest in the stock.

4. **Technical Indicators**:
   - **RSI**: Below 35 indicates oversold conditions, suggesting a rebound potential.
   - **Stochastic**: Below 25 confirms oversold levels.
   - **Bollinger Bands**: Price near the lower band increases the likelihood of upward movement.

5. **Known Stocks on an Uptrend**:
   - Focus on stocks you are familiar with and would not mind holding.
   - Identify stocks currently experiencing a pullback within a general uptrend.

6. **Premium Yield**:
   - Option premium should yield at least 1% of the stock price weekly.

---

## **Example Workflow**

### **Example: PLTR (Palantir Technologies)**
- Current price: $79/share.
- Sell a cash-secured put with a strike price of $75 (near 20 Delta).
- Expiration: Weekly (Friday).
- Premium: $1.00.
  - Yield = $1.00 / $79 = 1.27% weekly.
- If the stock price stays above $75, you keep the premium.
- If the stock price drops below $75, you buy the shares and start selling covered calls.

---

## **Future Enhancements**
1. **Add Dynamic Options Filtering**:
   - Fetch options chains to include premium yield and Delta in filtering.

2. **Integration with Charting Tools**:
   - Use libraries like `matplotlib` or APIs like TradingView for visual analysis.

3. **Enhanced Sector Analysis**:
   - Focus on specific sectors based on macroeconomic trends.

---

## **Contact**
For questions or support, please contact [Your Name/Email].

---

Let me know if further customizations or enhancements are required!

