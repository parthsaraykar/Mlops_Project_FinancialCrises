"""
MODIFIED STEP 0: Data Collection
Changes:
- Start date: 1990-01-01 (was 2005)
- Companies: 50 (was 25)
- Frequency: Quarterly (was weekly/daily)
"""

import pandas as pd
import numpy as np
import requests
import time
import calendar
import json
from pandas_datareader import data as pdr
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - MODIFIED
# =============================================================================

START_DATE = "1990-01-01"  # Changed from 2005
END_DATE = datetime.now().strftime("%Y-%m-%d")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS = ["XBAUMM6ATPHUYXTD"]
current_key_index = 0

def get_api_key():
    global current_key_index
    return API_KEYS[current_key_index % len(API_KEYS)]

def switch_api_key():
    global current_key_index
    current_key_index += 1

# =============================================================================
# EXPANDED COMPANIES - 100 TOTAL
# =============================================================================

COMPANIES = {
    # === FINANCIALS (18) ===
    "JPM":   {"name": "JPMorgan Chase",        "sector": "Financials"},
    "BAC":   {"name": "Bank of America",       "sector": "Financials"},
    "C":     {"name": "Citigroup",             "sector": "Financials"},
    "WFC":   {"name": "Wells Fargo",           "sector": "Financials"},
    "USB":   {"name": "U.S. Bancorp",          "sector": "Financials"},
    "PNC":   {"name": "PNC Financial",         "sector": "Financials"},
    "TFC":   {"name": "Truist Financial",      "sector": "Financials"},
    "BK":    {"name": "BNY Mellon",            "sector": "Financials"},
    "STT":   {"name": "State Street",          "sector": "Financials"},
    "GS":    {"name": "Goldman Sachs",         "sector": "Financials"},
    "MS":    {"name": "Morgan Stanley",        "sector": "Financials"},
    "SCHW":  {"name": "Charles Schwab",        "sector": "Financials"},
    "AXP":   {"name": "American Express",      "sector": "Financials"},
    "COF":   {"name": "Capital One",           "sector": "Financials"},
    "V":     {"name": "Visa",                  "sector": "Financials"},
    "MA":    {"name": "Mastercard",            "sector": "Financials"},
    "BRK.B": {"name": "Berkshire Hathaway",    "sector": "Financials"},
    "PGR":   {"name": "Progressive",           "sector": "Financials"},

    # === TECHNOLOGY (15) ===
    "AAPL":  {"name": "Apple",                "sector": "Technology"},
    "MSFT":  {"name": "Microsoft",            "sector": "Technology"},
    "IBM":   {"name": "IBM",                  "sector": "Technology"},
    "ORCL":  {"name": "Oracle",               "sector": "Technology"},
    "INTC":  {"name": "Intel",                "sector": "Technology"},
    "CSCO":  {"name": "Cisco",                "sector": "Technology"},
    "QCOM":  {"name": "Qualcomm",             "sector": "Technology"},
    "TXN":   {"name": "Texas Instruments",    "sector": "Technology"},
    "ADI":   {"name": "Analog Devices",       "sector": "Technology"},
    "AMZN":  {"name": "Amazon",               "sector": "Technology"},
    "GOOGL": {"name": "Alphabet",             "sector": "Technology"},
    "META":  {"name": "Meta",                 "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA",               "sector": "Technology"},
    "NFLX":  {"name": "Netflix",              "sector": "Technology"},
    "CRM":   {"name": "Salesforce",           "sector": "Technology"},

    # === ENERGY (10) ===
    "XOM":   {"name": "ExxonMobil",           "sector": "Energy"},
    "CVX":   {"name": "Chevron",              "sector": "Energy"},
    "COP":   {"name": "ConocoPhillips",       "sector": "Energy"},
    "SLB":   {"name": "Schlumberger",         "sector": "Energy"},
    "EOG":   {"name": "EOG Resources",        "sector": "Energy"},
    "PXD":   {"name": "Pioneer Natural",      "sector": "Energy"},
    "MPC":   {"name": "Marathon Petroleum",   "sector": "Energy"},
    "PSX":   {"name": "Phillips 66",          "sector": "Energy"},
    "VLO":   {"name": "Valero",               "sector": "Energy"},
    "OXY":   {"name": "Occidental Petroleum", "sector": "Energy"},

    # === HEALTHCARE (10) ===
    "JNJ":   {"name": "Johnson & Johnson",     "sector": "Healthcare"},
    "UNH":   {"name": "UnitedHealth",          "sector": "Healthcare"},
    "PFE":   {"name": "Pfizer",                "sector": "Healthcare"},
    "MRK":   {"name": "Merck",                 "sector": "Healthcare"},
    "ABBV":  {"name": "AbbVie",                "sector": "Healthcare"},
    "TMO":   {"name": "Thermo Fisher",         "sector": "Healthcare"},
    "ABT":   {"name": "Abbott Labs",           "sector": "Healthcare"},
    "LLY":   {"name": "Eli Lilly",             "sector": "Healthcare"},
    "BMY":   {"name": "Bristol Myers",         "sector": "Healthcare"},
    "AMGN":  {"name": "Amgen",                 "sector": "Healthcare"},

    # === CONSUMER STAPLES (10) ===
    "WMT":   {"name": "Walmart",              "sector": "Consumer Staples"},
    "PG":    {"name": "Procter & Gamble",     "sector": "Consumer Staples"},
    "KO":    {"name": "Coca-Cola",            "sector": "Consumer Staples"},
    "PEP":   {"name": "PepsiCo",              "sector": "Consumer Staples"},
    "COST":  {"name": "Costco",               "sector": "Consumer Staples"},
    "PM":    {"name": "Philip Morris",        "sector": "Consumer Staples"},
    "MO":    {"name": "Altria",               "sector": "Consumer Staples"},
    "CL":    {"name": "Colgate-Palmolive",    "sector": "Consumer Staples"},
    "MDLZ":  {"name": "Mondelez",             "sector": "Consumer Staples"},
    "KMB":   {"name": "Kimberly-Clark",       "sector": "Consumer Staples"},

    # === CONSUMER DISCRETIONARY (10) ===
    "HD":    {"name": "Home Depot",           "sector": "Consumer Discretionary"},
    "LOW":   {"name": "Lowes",                "sector": "Consumer Discretionary"},
    "MCD":   {"name": "McDonalds",            "sector": "Consumer Discretionary"},
    "NKE":   {"name": "Nike",                 "sector": "Consumer Discretionary"},
    "SBUX":  {"name": "Starbucks",            "sector": "Consumer Discretionary"},
    "TGT":   {"name": "Target",               "sector": "Consumer Discretionary"},
    "TJX":   {"name": "TJX Companies",        "sector": "Consumer Discretionary"},
    "GM":    {"name": "General Motors",       "sector": "Consumer Discretionary"},
    "F":     {"name": "Ford",                 "sector": "Consumer Discretionary"},
    "MAR":   {"name": "Marriott",             "sector": "Consumer Discretionary"},

    # === INDUSTRIALS (10) ===
    "BA":    {"name": "Boeing",               "sector": "Industrials"},
    "CAT":   {"name": "Caterpillar",          "sector": "Industrials"},
    "GE":    {"name": "General Electric",     "sector": "Industrials"},
    "HON":   {"name": "Honeywell",            "sector": "Industrials"},
    "MMM":   {"name": "3M",                   "sector": "Industrials"},
    "UPS":   {"name": "United Parcel Service", "sector": "Industrials"},
    "UNP":   {"name": "Union Pacific",        "sector": "Industrials"},
    "LMT":   {"name": "Lockheed Martin",      "sector": "Industrials"},
    "RTX":   {"name": "Raytheon Technologies","sector": "Industrials"},
    "DE":    {"name": "Deere & Company",      "sector": "Industrials"},

    # === COMMUNICATIONS (9) ===
    "DIS":   {"name": "Disney",               "sector": "Communications"},
    "CMCSA": {"name": "Comcast",              "sector": "Communications"},
    "VZ":    {"name": "Verizon",              "sector": "Communications"},
    "T":     {"name": "AT&T",                 "sector": "Communications"},
    "TMUS":  {"name": "T-Mobile",             "sector": "Communications"},
    "CHTR":  {"name": "Charter Communications","sector": "Communications"},
    "EA":    {"name": "Electronic Arts",      "sector": "Communications"},
    "ATVI":  {"name": "Activision Blizzard",  "sector": "Communications"},
    "TTWO":  {"name": "Take-Two Interactive", "sector": "Communications"},

    # === UTILITIES (5) ===
    "NEE":   {"name": "NextEra Energy",        "sector": "Utilities"},
    "DUK":   {"name": "Duke Energy",           "sector": "Utilities"},
    "SO":    {"name": "Southern Company",      "sector": "Utilities"},
    "D":     {"name": "Dominion Energy",       "sector": "Utilities"},
    "AEP":   {"name": "American Electric Power","sector": "Utilities"},

    # === REAL ESTATE (3) ===
    "AMT":   {"name": "American Tower",        "sector": "Real Estate"},
    "PLD":   {"name": "Prologis",              "sector": "Real Estate"},
    "SPG":   {"name": "Simon Property",        "sector": "Real Estate"},
}


# FRED and Market data unchanged
FRED_SERIES = {
    "GDPC1": "GDP",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment_Rate",
    "FEDFUNDS": "Federal_Funds_Rate",
    "T10Y3M": "Yield_Curve_Spread",
    "UMCSENT": "Consumer_Confidence",
    "DCOILWTICO": "Oil_Price",
    "BOPGSTB": "Trade_Balance",
    "BAA10Y": "Corporate_Bond_Spread",
    "TEDRATE": "TED_Spread",
    "DGS10": "Treasury_10Y_Yield",
    "STLFSI4": "Financial_Stress_Index",
    "BAMLH0A0HYM2": "High_Yield_Spread",
}

MARKET_TICKERS = {
    "^VIX": "VIX",
    "^GSPC": "SP500"
}

# =============================================================================
# YAHOO FINANCE API - MODIFIED FOR QUARTERLY
# =============================================================================

def yahoo_chart_api(ticker, start="1990-01-01", end=None, interval="3mo", timeout=20):
    """
    MODIFIED: Now fetches QUARTERLY data (interval="3mo")
    Was: interval="1wk" or "1d"
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    def to_unix(date_str):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return calendar.timegm(d.timetuple())

    start_ts = to_unix(start)
    end_ts = to_unix(end)
    
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history"
    )
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=timeout)
            
            if response.status_code != 200:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ValueError(f"HTTP {response.status_code}")
            
            data = response.json()
            
            if "chart" not in data or data["chart"].get("error"):
                raise ValueError("Invalid response")
            
            chart = data["chart"]["result"][0]
            ts = chart.get("timestamp", [])
            
            if not ts:
                raise ValueError("No timestamps")
            
            quote = chart["indicators"]["quote"][0]
            
            df = pd.DataFrame({
                "Date": pd.to_datetime(ts, unit="s"),
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            })
            
            if "adjclose" in chart["indicators"]:
                df["Adj_Close"] = chart["indicators"]["adjclose"][0]["adjclose"]
            else:
                df["Adj_Close"] = df["Close"]
            
            df = df.set_index("Date").dropna(how="any")
            
            if df.empty:
                raise ValueError("All data was NaN")
            
            return df
            
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise ValueError(f"Timeout after {timeout}s")
        
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise ValueError("Connection error")
        
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

    raise ValueError("Failed after 3 attempts")


# =============================================================================
# FALLBACK METHODS - MODIFIED FOR QUARTERLY
# =============================================================================

def fetch_with_yfinance(ticker, start, end, interval="3mo"):
    """MODIFIED: Changed default interval to quarterly"""
    try:
        import yfinance as yf
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError("Empty dataframe")
        
        df = df.reset_index()
        if 'Date' not in df.columns:
            df = df.rename(columns={'index': 'Date'})
        df = df.set_index('Date')
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        if 'Adj Close' in df.columns:
            df['Adj_Close'] = df['Adj Close']
        elif 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']]
        
    except ImportError:
        raise ValueError("yfinance library not installed")
    except Exception as e:
        raise ValueError(f"yfinance failed: {str(e)}")


def fetch_with_datareader(ticker, start, end):
    """Fallback - will resample to quarterly"""
    try:
        df = pdr.get_data_yahoo(ticker, start=start, end=end)
        
        if df.empty:
            raise ValueError("Empty dataframe")
        
        # Resample to quarterly if daily
        if len(df) > 200:  # Likely daily data
            df = df.resample('Q').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        if 'Adj Close' in df.columns:
            df['Adj_Close'] = df['Adj Close']
        elif 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']]
        
    except Exception as e:
        raise ValueError(f"datareader failed: {str(e)}")


# =============================================================================
# COMPANY PRICES - MODIFIED FOR QUARTERLY & 50 COMPANIES
# =============================================================================

def fetch_company_prices_raw():
    """
    MODIFIED: 
    - Now handles 50 companies (was 25)
    - Fetches quarterly data (was weekly)
    - Updated from 1990 (was 2005)
    """
    print("\n" + "=" * 70)
    print("STEP 3/4: FETCHING COMPANY PRICE DATA (QUARTERLY)")
    print("=" * 70)
    print(f"Companies: {len(COMPANIES)} (expanded from 25)")
    print(f"Frequency: Quarterly (changed from weekly/daily)")
    print(f"Start date: {START_DATE} (changed from 2005)")
    print()
    
    all_data = []
    failed_companies = []
    method_stats = {"ChartAPI": 0, "yfinance": 0, "datareader": 0}
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/50] {ticker:6} {info['name']:30}...", end=" ", flush=True)
        
        df = None
        method_used = None
        
        # Try Method 1: Yahoo Chart API (quarterly)
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "3mo")
            method_used = "ChartAPI"
        except Exception as e:
            print(f"Chart({str(e)[:20]})...", end=" ", flush=True)
        
        # Try Method 2: yfinance (quarterly)
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "3mo")
                method_used = "yfinance"
            except Exception as e:
                print(f"yf({str(e)[:20]})...", end=" ", flush=True)
        
        # Try Method 3: pandas_datareader (resample to quarterly)
        if df is None or df.empty:
            try:
                df = fetch_with_datareader(ticker, START_DATE, END_DATE)
                method_used = "datareader"
            except Exception as e:
                print(f"dr({str(e)[:20]})...", end=" ", flush=True)
        
        # Process successful fetch
        if df is not None and not df.empty:
            df = df.copy()
            df["Company"] = ticker
            df["Company_Name"] = info["name"]
            df["Sector"] = info["sector"]
            all_data.append(df)
            method_stats[method_used] += 1
            print(f"OK {len(df):,} quarters ({method_used})")
        else:
            print("FAILED all methods - SKIP")
            failed_companies.append(ticker)
        
        time.sleep(1)
    
    if not all_data:
        raise ValueError("ERROR: No company price data collected")
    
    if len(failed_companies) > 20:  # Allow more failures with 50 companies
        raise ValueError(f"ERROR: Too many failures ({len(failed_companies)}/50)")
    
    df_all = pd.concat(all_data, axis=0)
    df_all.index.name = 'Date'
    
    out = RAW_DIR / "company_prices_raw.csv"
    df_all.to_csv(out, index=True)
    
    print(f"\nSaved: {out}")
    print(f"Success: {len(all_data)}/{len(COMPANIES)} companies")
    
    if failed_companies:
        print(f"WARNING: Failed: {', '.join(failed_companies)}")
    else:
        print("SUCCESS: All 50 companies captured!")
    
    print(f"\nMethod Statistics:")
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(all_data)) * 100
            print(f"  {method:12}: {count:2d}/50 ({pct:5.1f}%)")
    
    return df_all


# =============================================================================
# OTHER FUNCTIONS UNCHANGED (FRED, Market, Fundamentals)
# =============================================================================

def fetch_fred_raw():
    """Unchanged - FRED data collection"""
    print("\n" + "=" * 70)
    print("STEP 1/4: FETCHING FRED MACROECONOMIC DATA")
    print("=" * 70)
    
    fred_data = {}
    successful = 0
    failed = []
    
    for series_id, col_name in FRED_SERIES.items():
        try:
            print(f"  {col_name:30} ({series_id})...", end=" ", flush=True)
            df = pdr.DataReader(series_id, "fred", START_DATE, END_DATE)
            fred_data[col_name] = df.iloc[:, 0]
            print(f"OK {len(df):,} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED ({str(e)[:40]})")
            failed.append(series_id)
    
    if not fred_data:
        raise ValueError("ERROR: No FRED data collected")
    
    df_fred = pd.DataFrame(fred_data)
    df_fred.index.name = 'Date'
    out = RAW_DIR / "fred_raw.csv"
    df_fred.to_csv(out)
    
    print(f"\nSaved: {out}")
    print(f"Success: {successful}/{len(FRED_SERIES)}")
    return df_fred


def fetch_market_raw():
    """Unchanged - Market data (VIX, S&P 500) - daily frequency"""
    print("\n" + "=" * 70)
    print("STEP 2/4: FETCHING MARKET DATA (DAILY)")
    print("=" * 70)
    
    market_data = {}
    successful = 0
    
    for ticker, name in MARKET_TICKERS.items():
        print(f"  {name:25} ({ticker})...", end=" ", flush=True)
        
        df = None
        method_used = None
        
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "1d")  # Keep daily for VIX/S&P
            method_used = "ChartAPI"
        except Exception as e:
            print(f"ChartAPI failed...", end=" ", flush=True)
        
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "1d")
                method_used = "yfinance"
            except:
                pass
        
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index.date)
            df.index.name = 'Date'
            market_data[name] = df["Close"]
            print(f"OK {len(df):,} records ({method_used})")
            successful += 1
        else:
            print("FAILED all methods")
        
        time.sleep(1)
    
    if not market_data:
        raise ValueError("ERROR: No market data collected")
    
    df_market = pd.DataFrame(market_data)
    df_market.index.name = 'Date'
    
    out = RAW_DIR / "market_raw.csv"
    df_market.to_csv(out)
    
    print(f"\nSaved: {out}")
    return df_market


# =============================================================================
# STEP 4: COMPANY FUNDAMENTALS - DIRECT FETCH (NO CACHE)
# =============================================================================

def fetch_company_fundamentals_raw():
    """
    Fetch company fundamentals directly from Alpha Vantage API.
    No caching - fetch fresh data every time like prices.
    """
    print("\n" + "=" * 70)
    print("STEP 4/4: COMPANY FUNDAMENTALS (QUARTERLY)")
    print("=" * 70)
    print(f"Companies: 50")
    print(f"Estimated time: ~{len(COMPANIES) * 40 / 60:.0f} minutes")
    print()
    
    all_income = []
    all_balance = []
    failed = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/50] {ticker:6} {info['name']:25}", end=" ", flush=True)
        
        # Fetch Income Statement
        income_data = fetch_alpha_vantage(ticker, "INCOME_STATEMENT")
        if income_data:
            df_inc = parse_financials(
                income_data,
                {
                    "Date": "fiscalDateEnding",
                    "Revenue": "totalRevenue",
                    "Net_Income": "netIncome",
                    "Gross_Profit": "grossProfit",
                    "Operating_Income": "operatingIncome",
                    "EBITDA": "ebitda",
                    "EPS": "reportedEPS",
                },
            )
            if not df_inc.empty:
                df_inc["Company"] = ticker
                df_inc["Company_Name"] = info["name"]
                df_inc["Sector"] = info["sector"]
                all_income.append(df_inc)
                print("Inc:OK", end=" ", flush=True)
            else:
                print("Inc:Empty", end=" ", flush=True)
        else:
            print("Inc:FAIL", end=" ", flush=True)
            failed.append(ticker)
        
        time.sleep(20)
        
        # Fetch Balance Sheet
        balance_data = fetch_alpha_vantage(ticker, "BALANCE_SHEET")
        if balance_data:
            df_bal = parse_financials(
                balance_data,
                {
                    "Date": "fiscalDateEnding",
                    "Total_Assets": "totalAssets",
                    "Total_Liabilities": "totalLiabilities",
                    "Total_Equity": "totalShareholderEquity",
                    "Current_Assets": "totalCurrentAssets",
                    "Current_Liabilities": "totalCurrentLiabilities",
                    "Long_Term_Debt": "longTermDebt",
                    "Short_Term_Debt": "shortTermDebt",
                    "Cash": "cashAndCashEquivalentsAtCarryingValue",
                },
            )
            if not df_bal.empty:
                df_bal["Company"] = ticker
                df_bal["Company_Name"] = info["name"]
                df_bal["Sector"] = info["sector"]
                
                # Calculate ratios
                df_bal["Debt_to_Equity"] = (
                    df_bal["Total_Liabilities"] / df_bal["Total_Equity"].replace(0, 1)
                )
                df_bal["Current_Ratio"] = (
                    df_bal["Current_Assets"] / df_bal["Current_Liabilities"].replace(0, 1)
                )
                
                all_balance.append(df_bal)
                print("Bal:OK")
            else:
                print("Bal:Empty")
        else:
            print("Bal:FAIL")
        
        time.sleep(20)
    
    # Save results
    print()
    
    if all_income:
        df_inc = pd.concat(all_income, ignore_index=True)
        df_inc.to_csv(RAW_DIR / "company_income_raw.csv", index=False)
        print(f"  ✓ Income saved: {len(all_income)} companies, {len(df_inc)} quarters")
    else:
        print("  WARNING: No income data collected")
    
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        df_bal.to_csv(RAW_DIR / "company_balance_raw.csv", index=False)
        print(f"  ✓ Balance saved: {len(all_balance)} companies, {len(df_bal)} quarters")
    else:
        print("  WARNING: No balance data collected")
    
    if failed:
        print(f"  WARNING: Failed companies: {', '.join(failed)}")
    
    print(f"\n✓ Fundamentals complete!")
    
    return (df_inc if all_income else None), (df_bal if all_balance else None)


def fetch_alpha_vantage(ticker, function, retry_count=0):
    """Fetch from Alpha Vantage with retry logic"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": get_api_key(),
        "datatype": "json"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        
        # Check for rate limiting
        if "Note" in data or "Information" in data:
            if retry_count < 3:
                switch_api_key()
                time.sleep(10)
                return fetch_alpha_vantage(ticker, function, retry_count + 1)
            return None
        
        if "quarterlyReports" not in data:
            return None
        
        return data["quarterlyReports"]
        
    except Exception as e:
        if retry_count < 2:
            time.sleep(5)
            return fetch_alpha_vantage(ticker, function, retry_count + 1)
        return None


def parse_financials(data, mapping):
    """Parse financial data"""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame([{k: r.get(v) for k, v in mapping.items()} for r in data])
    
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    return df


def main():
    """Main pipeline with modifications"""
    print("\n" + "=" * 70)
    print("MODIFIED FINANCIAL DATA LOADER")
    print("=" * 70)
    print(f"CHANGES:")
    print(f"  - Start date: 1990 (was 2005)")
    print(f"  - Companies: 50 (was 25)")
    print(f"  - Frequency: Quarterly (was weekly)")
    print(f"  - Fundamentals: Direct API fetch (no cache)")
    print("=" * 70)
    print(f"\n⚠️  ALPHA VANTAGE API KEY REQUIRED:")
    print(f"  - Current key: {API_KEYS[0]}")
    print(f"  - Free tier: 25 calls/day (need 100+ calls for 50 companies)")
    print(f"  - Consider: Premium key OR run over multiple days")
    print("=" * 70)
    
    overall_start = time.time()
    
    try:
        df_fred = fetch_fred_raw()
        df_market = fetch_market_raw()
        df_prices = fetch_company_prices_raw()
        df_income, df_balance = fetch_company_fundamentals_raw()
        
        elapsed = (time.time() - overall_start) / 60
        
        print("\n" + "=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f} minutes")
        print(f"\nModifications applied:")
        print(f"  ✓ 50 companies (expanded from 25)")
        print(f"  ✓ Quarterly data (changed from weekly/daily)")
        print(f"  ✓ Data from 1990 (extended from 2005)")
        
        print(f"\nFiles created in {RAW_DIR}/:")
        for f in sorted(RAW_DIR.glob("*.csv")):
            size = f.stat().st_size / (1024 * 1024)
            rows = sum(1 for _ in open(f)) - 1
            print(f"  {f.name:30} {size:>6.2f} MB  ({rows:>6,} rows)")
        
        print("=" * 70)
        print("Ready for Step 1: Data Cleaning!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()