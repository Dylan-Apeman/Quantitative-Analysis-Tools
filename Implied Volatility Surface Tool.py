import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Define a cleaning function to dynamically clean the options data
def clean_option_data(df):
    cleaned_df = df[(df['impliedVolatility'] > 0.01) &
                    (df['volume'] > 0) &
                    (df['openInterest'] > 0) &
                    (df['bid'] > 0) &
                    (df['ask'] > 0)]
    
    columns_to_drop = ['contractSymbol', 'lastTradeDate', 'currency', 'contractSize']
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    return cleaned_df

# Function to generate the 2D plots of IV vs Price
def plot_iv_vs_price(cleaned_calls_data, cleaned_puts_data, ticker):
    # Calculate mid-price for calls and puts
    cleaned_calls_data['price'] = (cleaned_calls_data['bid'] + cleaned_calls_data['ask']) / 2
    cleaned_puts_data['price'] = (cleaned_puts_data['bid'] + cleaned_puts_data['ask']) / 2
    
    # Convert IV to percentage
    cleaned_calls_data['impliedVolatility'] *= 100
    cleaned_puts_data['impliedVolatility'] *= 100

    # Group by price and calculate the mean implied volatility for calls and puts
    calls_grouped = cleaned_calls_data.groupby('price')['impliedVolatility'].mean().reset_index()
    puts_grouped = cleaned_puts_data.groupby('price')['impliedVolatility'].mean().reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot IV vs Price for Calls (line plot after aggregation)
    ax[0].plot(calls_grouped['price'], calls_grouped['impliedVolatility'], color='blue', label='Calls')
    ax[0].set_title(f"{ticker} Calls: IV vs. Price")
    ax[0].set_xlabel('Price')
    ax[0].set_ylabel('Implied Volatility (%)')

    # Plot IV vs Price for Puts (line plot after aggregation)
    ax[1].plot(puts_grouped['price'], puts_grouped['impliedVolatility'], color='red', label='Puts')
    ax[1].set_title(f"{ticker} Puts: IV vs. Price")
    ax[1].set_xlabel('Price')
    ax[1].set_ylabel('Implied Volatility (%)')

    plt.tight_layout()
    plt.show()


# Function to generate and display the plot in the Tkinter window
def generate_plot(ticker):
    try:
        stock = yf.Ticker(ticker)
        stockdata = stock.history(period="5d", interval="1m")
        dfs = pd.DataFrame(stockdata)
        dfs = dfs.drop(['Dividends', 'Stock Splits'], axis=1)

        # Display the stock history data in the terminal
        print("Stock History Data:")
        print(dfs.head())

        expirations = stock.options
        optiondata_list = []

        for expiry in expirations:
            opt_chain = stock.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            optiondata_list.append((calls, puts))

        calls_data = pd.concat([item[0] for item in optiondata_list], ignore_index=True)
        puts_data = pd.concat([item[1] for item in optiondata_list], ignore_index=True)

        calls_data['expiry'] = pd.to_datetime(calls_data['expiry'])
        puts_data['expiry'] = pd.to_datetime(puts_data['expiry'])

        current_date = datetime.now()
        calls_data['time_to_expiry'] = (calls_data['expiry'] - current_date).dt.days / 365
        puts_data['time_to_expiry'] = (puts_data['expiry'] - current_date).dt.days / 365

        # Dynamically clean the data using the defined function
        cleaned_calls_data = clean_option_data(calls_data)
        cleaned_puts_data = clean_option_data(puts_data)

        # Display the cleaned calls and puts data in the terminal
        print("\nCleaned Calls Data:")
        print(cleaned_calls_data.head())

        print("\nCleaned Puts Data:")
        print(cleaned_puts_data.head())

        # Polynomial regression for calls with degree 3
        X_calls = cleaned_calls_data[['strike', 'time_to_expiry']].values
        y_calls = cleaned_calls_data['impliedVolatility'].values
        poly_calls = PolynomialFeatures(degree=3)
        X_poly_calls = poly_calls.fit_transform(X_calls)
        reg_calls = LinearRegression()
        reg_calls.fit(X_poly_calls, y_calls)

        # Polynomial regression for puts with degree 3
        X_puts = cleaned_puts_data[['strike', 'time_to_expiry']].values
        y_puts = cleaned_puts_data['impliedVolatility'].values
        poly_puts = PolynomialFeatures(degree=3)
        X_poly_puts = poly_puts.fit_transform(X_puts)
        reg_puts = LinearRegression()
        reg_puts.fit(X_poly_puts, y_puts)

        # Generate strike and time grid for plotting
        strike_range_calls = np.linspace(cleaned_calls_data['strike'].min(), cleaned_calls_data['strike'].max(), 50)
        time_range_calls = np.linspace(cleaned_calls_data['time_to_expiry'].min(), cleaned_calls_data['time_to_expiry'].max(), 50)
        strike_grid_calls, time_grid_calls = np.meshgrid(strike_range_calls, time_range_calls)

        strike_range_puts = np.linspace(cleaned_puts_data['strike'].min(), cleaned_puts_data['strike'].max(), 50)
        time_range_puts = np.linspace(cleaned_puts_data['time_to_expiry'].min(), cleaned_puts_data['time_to_expiry'].max(), 50)
        strike_grid_puts, time_grid_puts = np.meshgrid(strike_range_puts, time_range_puts)

        # Predict implied volatility for calls and puts
        X_mesh_calls = poly_calls.transform(np.array([strike_grid_calls.ravel(), time_grid_calls.ravel()]).T)
        implied_vol_grid_calls = reg_calls.predict(X_mesh_calls).reshape(strike_grid_calls.shape)

        X_mesh_puts = poly_puts.transform(np.array([strike_grid_puts.ravel(), time_grid_puts.ravel()]).T)
        implied_vol_grid_puts = reg_puts.predict(X_mesh_puts).reshape(strike_grid_puts.shape)

        # Convert IV to percentage for the 3D surface plots
        implied_vol_grid_calls *= 100
        implied_vol_grid_puts *= 100

        # Create Matplotlib figure with two 3D surfaces
        fig = Figure(figsize=(14, 7))
        ax_calls = fig.add_subplot(1, 2, 1, projection='3d')
        ax_puts = fig.add_subplot(1, 2, 2, projection='3d')

        surf_calls = ax_calls.plot_surface(
            time_grid_calls, strike_grid_calls, implied_vol_grid_calls,
            cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.85
        )
        ax_calls.set_title(f"{ticker} Calls")
        ax_calls.set_xlabel('YTE')
        ax_calls.set_ylabel('K')
        ax_calls.set_zlabel('IV%')
        fig.colorbar(surf_calls, ax=ax_calls, shrink=0.6, pad=0.1, label='IV% (Calls)')

        surf_puts = ax_puts.plot_surface(
            time_grid_puts, strike_grid_puts, implied_vol_grid_puts,
            cmap=cm.cividis, linewidth=0, antialiased=False, alpha=0.85
        )
        ax_puts.set_title(f"{ticker} Puts")
        ax_puts.set_xlabel('YTE')
        ax_puts.set_ylabel('K')
        ax_puts.set_zlabel('IV%')
        fig.colorbar(surf_puts, ax=ax_puts, shrink=0.6, pad=0.1, label='IV% (Puts)')

        fig.suptitle(f"Volatility Surface of {ticker} Options")
        fig.tight_layout()

        global figure_canvas
        if figure_canvas is not None:
            figure_canvas.get_tk_widget().destroy()

        figure_canvas = FigureCanvasTkAgg(fig, master=figure_frame)
        figure_canvas.draw()
        figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Plot the IV vs Price graphs for Calls and Puts
        plot_iv_vs_price(cleaned_calls_data, cleaned_puts_data, ticker)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Set up the Tkinter window
root = tk.Tk()
root.title("Volatility Surface")

# Input field for the ticker
label = tk.Label(root, text="Enter Ticker Symbol:")
label.pack()

ticker_entry = tk.Entry(root)
ticker_entry.pack()

# Button to generate the plot
button = tk.Button(root, text="Generate", command=lambda: generate_plot(ticker_entry.get()))
button.pack()

# Frame to host the Matplotlib figure
figure_frame = tk.Frame(root)
figure_frame.pack(fill=tk.BOTH, expand=True)
figure_canvas = None

# Start the Tkinter event loop
root.mainloop()
