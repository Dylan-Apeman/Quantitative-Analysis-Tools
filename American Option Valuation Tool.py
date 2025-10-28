import tkinter as tk
import math
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    expiry_dates = stock.options
    if not expiry_dates:
        return None, None
    return stock.option_chain, expiry_dates

def rfrate():
    irate = 4.508
    XIIIw = yf.Ticker('^IRX').history(period="1d")['Close'].iloc[-1]
    Vy = yf.Ticker('^FVX').history(period="1d")['Close'].iloc[-1]
    Xy = yf.Ticker('^TNX').history(period="1d")['Close'].iloc[-1]
    XXXy = yf.Ticker('^TYX').history(period="1d")['Close'].iloc[-1]
    return max(XIIIw, Vy, Xy, XXXy) - irate

def find_closest_expiry(desired_days, expiry_dates):
    today = datetime.today()
    desired_expiry_date = today + timedelta(days=desired_days)
    closest_expiry = min(expiry_dates, key=lambda x: abs(pd.to_datetime(x) - desired_expiry_date))
    return closest_expiry

def calculate_historical_volatility(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    historical_data = stock.history(start=start_date, end=end_date)
    daily_returns = historical_data['Close'].pct_change().dropna()
    volatility = np.std(daily_returns) * np.sqrt(252)
    return volatility

def determine_number_of_steps(days_to_expiry):
    steps = int(max(150, round(100 * math.sqrt(days_to_expiry))))
    return steps

def american_option_binomial_tree(S, K, T, r, sigma, N, option_type='call'):
    if sigma < 0.001 or T < 0.01:
        if option_type == 'call':
            return max(S - K, 0)
        elif option_type == 'put':
            return max(K - S, 0)

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    asset_prices = np.zeros(N + 1)
    option_values = np.zeros(N + 1)

    for i in range(N + 1):
        asset_prices[i] = S * (u ** i) * (d ** (N - i))

    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == 'put':
        option_values = np.maximum(0, K - asset_prices)

    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            asset_prices[i] = asset_prices[i] * d
            option_values[i] = (q * option_values[i + 1] + (1 - q) * option_values[i]) * np.exp(-r * dt)
            if option_type == 'call':
                option_values[i] = np.maximum(option_values[i], asset_prices[i] - K)
            elif option_type == 'put':
                option_values[i] = np.maximum(option_values[i], K - asset_prices[i])

    return option_values[0]

def price_american_option_from_yfinance(ticker, option_type, strike_price, desired_days_to_expiry):
    option_chain, expiry_dates = fetch_options_data(ticker)
    if option_chain is None or expiry_dates is None:
        raise ValueError(f"No options data available for ticker {ticker}")

    closest_expiry = find_closest_expiry(desired_days_to_expiry, expiry_dates)
    sigma = calculate_historical_volatility(ticker)
    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    T = (pd.to_datetime(closest_expiry) - pd.Timestamp.today()).days / 365
    r = rfrate()
    N = determine_number_of_steps(desired_days_to_expiry)
    option_price = american_option_binomial_tree(current_price, strike_price, T, r, sigma, N, option_type)
    
    return option_price

# Tkinter GUI
def submit():
    ticker = entry_ticker.get().upper()
    option_type = option_type_var.get().lower()
    strike_price = float(entry_strike_price.get())
    days_to_expiry = int(entry_days_to_expiry.get())

    try:
        option_price = price_american_option_from_yfinance(ticker, option_type, strike_price, days_to_expiry)
        result_label.config(text=f"The price of the American option is: ${option_price:.2f}")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# Create Tkinter window
root = tk.Tk()
root.title("American Option Pricing")

# Define window size
window_width = 400
window_height_initial = 300

# Get the screen's width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position for the window to be centered
position_top = int((screen_height - window_height_initial) / 2)
position_right = int((screen_width - window_width) / 2)

# Set the window size and position it in the center of the screen
root.geometry(f"{window_width}x{window_height_initial}+{position_right}+{position_top}")

# Center the elements using grid and padding
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Create input fields
tk.Label(root, text="Stock Ticker").grid(row=0, column=0, pady=10, sticky="e")
entry_ticker = tk.Entry(root)
entry_ticker.grid(row=0, column=1, padx=10)

tk.Label(root, text="Option Type").grid(row=1, column=0, pady=10, sticky="e")
option_type_var = tk.StringVar(value="call")
option_menu = tk.OptionMenu(root, option_type_var, "call", "call", "put")
option_menu.grid(row=1, column=1, padx=10)

tk.Label(root, text="Strike Price").grid(row=2, column=0, pady=10, sticky="e")
entry_strike_price = tk.Entry(root)
entry_strike_price.grid(row=2, column=1, padx=10)

tk.Label(root, text="Days to Expiry").grid(row=3, column=0, pady=10, sticky="e")
entry_days_to_expiry = tk.Entry(root)
entry_days_to_expiry.grid(row=3, column=1, padx=10)

# Submit button
submit_button = tk.Button(root, text="Calculate", command=submit)
submit_button.grid(row=4, columnspan=2, pady=10)

# Label to display the result
result_label = tk.Label(root, text="")
result_label.grid(row=5, columnspan=2, pady=10)

# Label to display more calculations (like volatility, stock price, expiry date)
details_label = tk.Label(root, text="")
details_label.grid(row=6, columnspan=2, pady=10)

root.mainloop()