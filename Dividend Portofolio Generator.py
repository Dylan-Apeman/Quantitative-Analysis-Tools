import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Example list of S&P 500 companies
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'BRK.B', 'META', 'XOM', 'UNH', 'JNJ', 'V', 'JPM', 'PG', 'LLY', 
           'MA', 'HD', 'MRK', 'PEP', 'ABBV', 'KO', 'PFE', 'CVX', 'AVGO', 'ADBE', 'COST', 'TMO', 'NFLX', 'MCD', 'ABT', 'DIS', 
           'CRM', 'TXN', 'NEE', 'LIN', 'NKE', 'WMT', 'DHR', 'BMY', 'AMD', 'QCOM', 'LOW', 'SPGI', 'PM', 'INTC', 'MDT', 'AMGN', 
           'HON', 'RTX', 'CAT', 'AXP', 'BA', 'GS', 'MS', 'USB', 'UNP', 'ZTS', 'SCHW', 'CVS', 'AMT', 'PLD', 'BLK', 'T', 'DE', 
           'ISRG', 'SYK', 'COP', 'C', 'LMT', 'MMC', 'CI', 'NOW', 'SCHW', 'BKNG', 'ADP', 'TMUS', 'IBM', 'VRTX', 'EW', 'PYPL', 
           'FISV', 'GILD', 'SBUX', 'CB', 'MDLZ', 'ELV', 'BDX', 'PGR', 'CME', 'MO', 'TFC', 'ITW', 'ADI', 'TGT', 'SO', 'EQIX', 
           'DUK', 'SLB', 'PSA', 'APD', 'F', 'LRCX', 'HCA', 'GM', 'NOC', 'CNC', 'AON', 'WM', 'CL', 'ICE', 'KMB', 'EOG', 'ATVI', 
           'REGN', 'ORCL', 'ILMN', 'ROST', 'ETSY', 'STZ', 'EA', 'HPQ', 'CTVA', 'WBA', 'SPG', 'BIIB', 'TWTR', 'DOW', 'BBY', 
           'VFC', 'HPE', 'NWSA', 'FBHS', 'KMI', 'BWA', 'TSN']

# Function to get dividend yield for each stock in S&P 500
def get_dividend_yield(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    dividend_yield = info.get('dividendYield', None)  # dividendYield is a fraction
    return dividend_yield if dividend_yield is not None else 0

# Step 2: Retrieve dividend yield for each stock
dividend_yields = []
for ticker in tickers:
    try:
        yield_ = get_dividend_yield(ticker)
        dividend_yields.append((ticker, yield_))
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        dividend_yields.append((ticker, 0))

# Step 3: Sort the list by dividend yield and display top 10 stocks
dividend_yields_sorted = sorted(dividend_yields, key=lambda x: x[1], reverse=True)
df_dividend_yields = pd.DataFrame(dividend_yields_sorted, columns=['Ticker', 'Dividend Yield'])
top_dividend_stocks = df_dividend_yields.head(10)
print("\nTop 10 S&P 500 stocks by dividend yield:")
print(top_dividend_stocks)

# Secondary Functionality: Estimate next ex-dividend date and expected dividend value
def estimate_next_ex_dividend(ticker):
    stock = yf.Ticker(ticker)
    try:
        dividend_data = stock.dividends
        if dividend_data.empty:
            print(f"No dividend history available for {ticker}.")
            return None, None, None

        # Estimate the expected dividend value as the average of the last four dividends
        recent_dividends = dividend_data[-4:]
        avg_dividend = recent_dividends.mean()

        # Calculate the average interval between ex-dividend dates
        dividend_dates = recent_dividends.index.to_list()
        intervals = [(dividend_dates[i] - dividend_dates[i - 1]).days for i in range(1, len(dividend_dates))]
        avg_interval = sum(intervals) / len(intervals)
        estimated_next_ex_date = dividend_dates[-1] + timedelta(days=avg_interval)

        # Calculate the dividend yield based on the last close price
        last_close = stock.history(period="1d")["Close"].iloc[0]
        dividend_yield = (avg_dividend / last_close) * 100

        return estimated_next_ex_date.date(), avg_dividend, dividend_yield
    except Exception as e:
        print(f"Error estimating dividend data for {ticker}: {e}")
        return None, None, None

# User input for secondary functionality
ticker_input = input("\nEnter a stock ticker symbol for ex-dividend analysis: ").upper()
ex_date, dividend_value, yield_percent = estimate_next_ex_dividend(ticker_input)

if ex_date and dividend_value and yield_percent:
    print(f"\nEstimated dividend data for {ticker_input}:")
    print(f"  Next ex-dividend date: {ex_date}")
    print(f"  Expected dividend value per share: ${dividend_value:.2f}")
    print(f"  Estimated dividend yield: {yield_percent:.2f}%")
else:
    print(f"Could not estimate dividend data for {ticker_input}.")