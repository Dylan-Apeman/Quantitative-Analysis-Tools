import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate mu and sigma from historical stock data
def get_mu_sigma(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()  # Dropping NaN values after pct_change
    mu = data['Returns'].mean() * 252  # Annualized mean return
    sigma = data['Returns'].std() * np.sqrt(252)  # Annualized volatility
    return mu, sigma

# Function to simulate stock prices using Geometric Brownian Motion
def simulate_stock_prices(S0, mu, sigma, T, steps):
    dt = T / steps
    prices = np.zeros(steps + 1)  # Ensure we start with initial price
    prices[0] = S0
    for t in range(1, steps + 1):  # Simulate over steps+1 to include S0
        z = np.random.normal(0, 1)
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

# Payoff function for a covered call
def covered_call_payoff(S_T, K, premium, S0):
    long_stock = S_T - S0  # Payoff from holding the stock
    short_call = -max(S_T - K, 0) + premium  # Payoff from writing the call
    return long_stock + short_call

# Get user inputs
ticker = input("Enter the stock ticker (e.g., AAPL, MSFT): ").upper()
S0 = float(input("Enter the current stock price (S0): "))
K = float(input("Enter the strike price of the call option (K): "))
premium = float(input("Enter the premium received from selling the call option: "))
T = float(input("Enter days to expiration (DTE): ")) / 365.25
steps = 1000  # Number of steps for simulation
num_simulations = 10000  # Number of Monte Carlo simulations

# Calculate mu and sigma for the selected ticker
mu, sigma = get_mu_sigma(ticker)

# Simulating the stock price at expiration using GBM
stock_prices_at_T = []
for _ in range(num_simulations):
    prices = simulate_stock_prices(S0, mu, sigma, T, steps)
    stock_prices_at_T.append(prices[-1])

# Calculate the payoff for each simulated price
payoffs = [covered_call_payoff(S_T, K, premium, S0) for S_T in stock_prices_at_T]

# Plotting the payoff diagram with a wider range of stock prices
stock_price_range = np.linspace(S0 * 0.1, S0 * 2, 500)  # Create a wider range for stock prices at expiration
payoff_curve = [covered_call_payoff(S_T, K, premium, S0) for S_T in stock_price_range]

plt.figure(figsize=(10,6))
plt.plot(stock_price_range, payoff_curve, label="Covered Call Payoff")
plt.axvline(x=K, linestyle="--", color="gray", label="Strike Price")
plt.title(f"Covered Call Payoff Diagram for {ticker}")
plt.xlabel("Stock Price at Expiration")
plt.ylabel("Payoff")
plt.legend()
plt.grid(True)
plt.show()

print(f"Calculated mu (annualized return): {mu:.2f}")
print(f"Calculated sigma (annualized volatility): {sigma:.2f}")
