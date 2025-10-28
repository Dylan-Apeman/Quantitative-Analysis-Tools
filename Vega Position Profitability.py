import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.interpolate import griddata


# Default start and end dates for historical data
enddate = datetime.now()
startdate = enddate - timedelta(days = 252)

# Fetch historical stock data
def get_data(ticker, start=startdate, end=enddate):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    return data[['Close', 'Volume', 'Log_Returns']]

# Dynamically find the most recent earnings date and predict the next one
def get_recent_and_next_earnings(ticker):
    stock = yf.Ticker(ticker)
    earnings_dates = stock.earnings_dates

    if earnings_dates is None or earnings_dates.empty:
        return None, None
    
    most_recent_earnings = earnings_dates.index[0].tz_localize(None)
    next_earnings = most_recent_earnings + relativedelta(months=3)

    if next_earnings <= datetime.now():
        next_earnings = datetime.now() + relativedelta(months=3)

    return most_recent_earnings, next_earnings

# Perform GARCH analysis
def garch_analysis(data):
    model = arch_model(data['Log_Returns'].dropna(), vol='Garch', p=1, q=1)
    fitted = model.fit(disp="off")
    return fitted

# Mean-reversion test (Augmented Dickey-Fuller test)
def mean_reversion_test(data):
    test_stat, p_value, *_ = adfuller(data.dropna())
    return p_value < 0.05  # True if mean-reversion detected

# Plot Vega Exposure and Data Trends
def plot_vega_exposure(data, ticker, vega, iv_changes):
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    # Vega exposure plot
    ax[0].plot(iv_changes, vega * np.array(iv_changes), label="Vega Exposure")
    ax[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax[0].set_title("Vega Exposure vs. IV Changes")
    ax[0].set_xlabel("IV Changes")
    ax[0].set_ylabel("Vega Exposure")
    ax[0].legend()

    # Historical price and returns
    ax[1].plot(data.index, data['Close'], label="Stock Price")
    ax[1].set_title(f"Historical Stock Prices - {ticker}")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Price")
    ax[1].legend()

    plt.tight_layout()
    return fig

# Main function for analysis
def analyze_vega_position(ticker):
    results = {}

    try:
        # Fetch data
        data = get_data(ticker)
        results["Most Recent Earnings"], results["Next Earnings"] = get_recent_and_next_earnings(ticker)

        # GARCH Analysis
        garch_result = garch_analysis(data)
        results["GARCH Summary"] = garch_result.summary().as_text()

        # Mean-Reversion Test
        results["Mean Reversion"] = mean_reversion_test(data['Log_Returns'])

        # Vega Exposure Plot
        vega = 0.01  # Example vega for demonstration
        iv_changes = np.linspace(-0.2, 0.2, 100)
        fig = plot_vega_exposure(data, ticker, vega, iv_changes)

        return results, fig
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while analyzing {ticker}: {e}")
        return None, None


def plot_3d_options(ticker, frame):
    # Fetch stock data
    stock = yf.Ticker(ticker)
    
    # Get all expiration dates
    try:
        expiration_dates = stock.options
        if not expiration_dates:
            messagebox.showerror("Data Error", f"No options data available for {ticker}.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Unable to fetch options expiration dates: {e}")
        return

    # Current date for DTE calculation
    current_date = datetime.now()

    # Lists to store combined data
    all_strikes, all_ivs, all_dtes = [], [], []

    # Loop through expiration dates to fetch options data
    for exp_date in expiration_dates:
        options_chain = stock.option_chain(exp_date)
        calls = options_chain.calls

        # Calculate DTE
        dte = (pd.to_datetime(exp_date) - current_date).days

        # Append data
        all_strikes.extend(calls['strike'])
        all_ivs.extend(calls['impliedVolatility'] * 100)  # Convert to percentage
        all_dtes.extend([dte] * len(calls))  # Use DTE instead of raw expiration dates

    # Convert data to numpy arrays for processing
    all_strikes = np.array(all_strikes)
    all_ivs = np.array(all_ivs)
    all_dtes = np.array(all_dtes)

    # Create a grid for smooth interpolation
    grid_strikes = np.linspace(min(all_strikes), max(all_strikes), 50)
    grid_dtes = np.linspace(min(all_dtes), max(all_dtes), 50)
    strike_grid, dte_grid = np.meshgrid(grid_strikes, grid_dtes)

    # Interpolate implied volatilities for the grid
    iv_grid = griddata((all_strikes, all_dtes), all_ivs, (strike_grid, dte_grid), method='cubic')

    # Create the 3D plot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the volatility surface
    surface = ax.plot_surface(strike_grid, dte_grid, iv_grid, cmap='viridis', alpha=0.8, edgecolor='none')

    # Set titles and labels
    ax.set_title(f"{ticker} Volatility Surface")
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Days to Expiry (DTE)")
    ax.set_zlabel("Implied Volatility (IV%)")

    # Add color bar for the surface
    cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label("Implied Volatility (%)")

    # Embed the plot into the frame
    for widget in frame.winfo_children():
        widget.destroy()  # Clear any existing widgets in the frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


def main_gui():
    def analyze():
        ticker = ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showerror("Input Error", "Please enter a valid stock ticker.")
            return

        # Perform Vega position analysis
        results, fig = analyze_vega_position(ticker)

        if results:
            # Display results in the text area
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Most Recent Earnings: {results['Most Recent Earnings']}\n")
            result_text.insert(tk.END, f"Next Earnings: {results['Next Earnings']}\n")
            result_text.insert(tk.END, "GARCH Summary:\n")
            result_text.insert(tk.END, results["GARCH Summary"])
            result_text.insert(tk.END, f"\nMean-Reversion Detected: {results['Mean Reversion']}\n")

            # Embed the Vega exposure plot
            for widget in plot_frame.winfo_children():
                widget.destroy()  # Clear existing plots
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            # Embed the 3D plot
            plot_3d_options(ticker, plot_3d_frame)

    # Tkinter window
    window = tk.Tk()
    window.title("Vega Position Analysis")
    window.state("zoomed")

    # Configure grid layout
    window.grid_rowconfigure(1, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)

    # Input Frame
    input_frame = ttk.Frame(window)
    input_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    ttk.Label(input_frame, text="Stock Ticker:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ticker_entry = ttk.Entry(input_frame, width=10)
    ticker_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

    analyze_button = ttk.Button(input_frame, text="Analyze", command=analyze)
    analyze_button.grid(row=0, column=2, sticky="w", padx=5, pady=5)

    # Results Frame
    results_frame = ttk.Frame(window, relief=tk.SUNKEN, borderwidth=1)
    results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    results_frame.grid_rowconfigure(0, weight=1)
    results_frame.grid_columnconfigure(0, weight=1)

    result_text = tk.Text(results_frame, wrap=tk.WORD, height=20)
    result_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Plot Frame for Vega exposure
    plot_frame = ttk.Frame(window, relief=tk.SUNKEN, borderwidth=1)
    plot_frame.grid(row=1, column=1, rowspan = 2, sticky="nsew", padx=10, pady=10)
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    # Plot Frame for 3D options plot
    plot_3d_frame = ttk.Frame(window, relief=tk.SUNKEN, borderwidth=1)
    plot_3d_frame.grid(row=2, column=0, columnspan=1, sticky="nsew", padx=10, pady=10)
    plot_3d_frame.grid_rowconfigure(0, weight=1)
    plot_3d_frame.grid_columnconfigure(0, weight=1)

    window.mainloop()

if __name__ == "__main__":
    main_gui()
