import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Fetch historical data for tickers
def fetch_data(tickers):
    today = datetime.now()
    data = yf.download(tickers, start=today-timedelta(days=730), end=today)['Close']
    return data

def get_risk_free_rate():
    # Fetch data for TIP
    tip_data = yf.Ticker("TIP")
    # Get the dividend yield (annual) as a proxy for the risk-free rate
    tip_info = tip_data.info
    risk_free_rate = tip_info.get("yield", 0)  # Default to 0 if yield is not available
    return risk_free_rate

# Calculate expected returns and covariance matrix
def calculate_returns_and_cov(data, bond_tickers=[]):
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = pd.Series(index=data.columns)
    cov_matrix = returns.cov() * 252  # Annualize covariance matrix for all assets
    for ticker in data.columns:
        if ticker in bond_tickers:
            # Fetch current yield for the bond (use as the expected return)
            bond_info = yf.Ticker(ticker).info
            current_yield = bond_info.get("yield", 0)  # If yield not available, use 0
            mean_returns[ticker] = current_yield
        else:
            # For stocks, calculate the expected return from historical prices
            mean_returns[ticker] = returns[ticker].mean() * 252  # Annualized return for stock
    return mean_returns, cov_matrix

# Define portfolio optimization using Modern Portfolio Theory
def portfolio_optimization(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    
    # Objective function (minimize negative Sharpe ratio)
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
        return -sharpe_ratio

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess
    init_guess = num_assets * [1. / num_assets]
    
    # Optimize for the best weights
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # Return the optimal weights

def calculate_shares(weights, tickers, portfolio_value, bond_tickers=[]):
    current_prices = yf.download(tickers, period="1d")['Close'].iloc[-1]
    # Manually set bond prices to $100
    bond_prices = pd.Series(100, index=bond_tickers) if bond_tickers else pd.Series()
    # Replace bond prices in the downloaded prices
    current_prices.update(bond_prices)
    # Calculate allocation for each asset
    allocation = weights * portfolio_value
    # Calculate number of shares (for bonds, this will be the number of $100 units)
    shares_to_buy = allocation // current_prices.values
    # Calculate total price for each asset (shares * current price)
    total_prices = shares_to_buy * current_prices.values
    # Return the number of shares, current prices, and total prices
    return shares_to_buy, current_prices.values, total_prices

def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.01, num_points=100):
    num_assets = len(mean_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)
    
    # Minimize volatility for each return target
    frontier_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    frontier_volatility = []
    
    for target_return in frontier_returns:
        # Define constraints and bounds
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
            {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}  # Target return
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        result = minimize(portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        frontier_volatility.append(result.fun)
    
    return frontier_volatility, frontier_returns

def plot_efficient_frontier(mean_returns, cov_matrix, weights, risk_free_rate=0.01):
    # Calculate the efficient frontier
    frontier_volatility, frontier_returns = calculate_efficient_frontier(mean_returns, cov_matrix)
    
    # Calculate the optimal portfolio's return and risk (standard deviation)
    portfolio_return_opt = np.dot(weights, mean_returns)
    portfolio_stddev_opt = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Plot the efficient frontier
    plt.figure(figsize=(10,6))
    plt.plot(frontier_volatility, frontier_returns, 'k-', label='Efficient Frontier')  # Black line for efficient frontier
    plt.scatter(portfolio_stddev_opt, portfolio_return_opt, c='green', marker='o', s=100, label='Optimal Portfolio')  # Green dot for optimal portfolio
    
    # Labels and title
    plt.title('Efficient Frontier with Optimal Portfolio')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

# Main function to run the script
def main():
    # Get user input for stocks and bonds
    stock_tickers_input = input("Enter a list of stock tickers separated by commas: ")
    bond_tickers_input = input("Enter a list of bond tickers separated by commas (optional): ")
    portfolio_value = float(input("Enter the total value of the portfolio in USD: "))
    
    # Process tickers for stocks and bonds
    stock_tickers = [ticker.strip().upper() for ticker in stock_tickers_input.split(',')]
    bond_tickers = [ticker.strip().upper() for ticker in bond_tickers_input.split(',') if ticker]  # Filter out empty tickers
    
    # Fetch data and calculate returns and covariance for stocks
    stock_data = fetch_data(stock_tickers)
    
    # Fetch data and calculate returns and covariance for bonds (if provided)
    if bond_tickers:
        bond_data = fetch_data(bond_tickers)

        # Convert both stock_data and bond_data to tz-naive before concatenating
        stock_data = stock_data.tz_localize(None)
        bond_data = bond_data.tz_localize(None)
        
        all_data = pd.concat([stock_data, bond_data], axis=1)
        all_mean_returns, all_cov_matrix = calculate_returns_and_cov(all_data, bond_tickers=bond_tickers)
        tickers = stock_tickers + bond_tickers  # Combine ticker list for printing
    else:
        # If no bonds, use only stock data
        all_mean_returns, all_cov_matrix = calculate_returns_and_cov(stock_data)
        tickers = stock_tickers

    # Get the risk-free rate from the TIP ETF
    risk_free_rate = get_risk_free_rate()
    print(f"Using risk-free rate: {risk_free_rate:.2%}")

    # Optimize the portfolio weights using MPT
    optimal_weights = portfolio_optimization(all_mean_returns, all_cov_matrix, risk_free_rate)
    
    # Calculate the number of shares to buy for each asset (stocks and bonds)
    shares_to_buy = calculate_shares(optimal_weights, tickers, portfolio_value)
    
    # Calculate the number of shares to buy for each asset (stocks and bonds)
    shares_to_buy, current_prices, total_prices = calculate_shares(optimal_weights, tickers, portfolio_value, bond_tickers)

    # Output results
    print("\nOptimal Weights:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {optimal_weights[i]:.2%}")

    print("\nNumber of Shares to Buy, Current Price, and Total Price:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {int(shares_to_buy[i])} shares, Current Price: ${current_prices[i]:.2f}, Total Price: ${total_prices[i]:,.2f}")

    # Plot the efficient frontier with the calculated portfolio highlighted
    plot_efficient_frontier(all_mean_returns, all_cov_matrix, optimal_weights, risk_free_rate)


if __name__ == "__main__":
    main()
