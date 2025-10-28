# 📈 American Options and Portfolio Tools — Calvin Lomax

This repository contains a collection of **Python-based quantitative finance tools** built by **Calvin Lomax** for research and portfolio management.  
Each script focuses on a specific aspect of **options pricing, portfolio construction, or market analytics**, leveraging **`yfinance`**, **`NumPy`**, **`Pandas`**, and visualization libraries.

---

## 🧮 Tools Overview

### 1. **American Option Valuation Tool**  
**File:** `American Option Valuation Tool.py`  
A Tkinter GUI application for pricing **American call and put options** using a **binomial tree model**.

**Features:**  
- Fetches real-time data via `yfinance`.  
- Automatically selects the nearest expiry date.  
- Calculates historical volatility and risk-free rate dynamically.  
- Provides an interactive GUI for input and results display.  

**Key Functions:**  
- `price_american_option_from_yfinance()`  
- `american_option_binomial_tree()`  
- `rfrate()`  

---

### 2. **Covered Call Modeling Tool**  
**File:** `Covered Call Modeling Tool.py`  
Simulates **covered call strategies** using **Monte Carlo simulation** and **geometric Brownian motion**.

**Features:**  
- Estimates expected return (`mu`) and volatility (`sigma`) from historical data.  
- Simulates future price paths and payoff outcomes.  
- Visualizes the payoff diagram using Matplotlib.  

**Core Functions:**  
- `get_mu_sigma()`  
- `simulate_stock_prices()`  
- `covered_call_payoff()`  

---

### 3. **Dividend Portfolio Generator**  
**File:** `Dividend Portofolio Generator.py`  
Generates a **high-dividend portfolio** from S&P 500 tickers and estimates **next ex-dividend dates**.

**Features:**  
- Fetches dividend yields for all S&P 500 components.  
- Ranks top 10 by yield.  
- Estimates next ex-dividend dates and dividend per share using historical intervals.  

**Core Functions:**  
- `get_dividend_yield()`  
- `estimate_next_ex_dividend()`  

---

### 4. **Implied Volatility Surface Tool**  
**File:** `Implied Volatility Surface Tool.py`  
A Tkinter-based visualization of **3D implied volatility surfaces** for both calls and puts.

**Features:**  
- Pulls full option chains from `yfinance`.  
- Cleans and filters illiquid contracts.  
- Fits polynomial regressions (degree 3) to smooth the IV surface.  
- Generates both 2D and 3D volatility plots.  

**Core Functions:**  
- `clean_option_data()`  
- `generate_plot()`  
- `plot_iv_vs_price()`  

---

### 5. **Poor Man’s Covered Call Generator**  
**File:** `Poor Man's CC Generator.py`  
Automates **LEAPS-based covered call** (PMCC) construction.

**Features:**  
- Identifies optimal LEAPS (long-term ITM) and short-term OTM call pairings.  
- Filters valid bid-ask combinations.  
- Computes the **net premium** and displays optimal pairing.  

**Core Functions:**  
- `get_options_chain()`  
- `filter_valid_options()`  
- `pmcc_strategy()`  

---

### 6. **Portfolio Allocation Tool**  
**File:** `Portfolio Allocation Tool.py`  
Implements **Modern Portfolio Theory (MPT)** for optimal asset allocation.

**Features:**  
- Calculates expected returns and covariance matrices.  
- Optimizes portfolio weights via **Sharpe ratio maximization**.  
- Plots the **efficient frontier**.  
- Supports stock and bond inputs with dynamic risk-free rate estimation.  

**Core Functions:**  
- `portfolio_optimization()`  
- `calculate_returns_and_cov()`  
- `plot_efficient_frontier()`  

---

### 7. **Vega Position Profitability Tool**  
**File:** `Vega Position Profitability.py`  
Performs **volatility exposure analysis** and **GARCH modeling** for Vega-sensitive positions.

**Features:**  
- Performs **GARCH(1,1)** volatility modeling.  
- Tests for mean reversion using **ADF test**.  
- Visualizes Vega exposure vs. implied volatility changes.  
- Displays a 3D implied volatility surface from live option data.  
- Built with a **Tkinter GUI** for interactive exploration.  

**Core Functions:**  
- `analyze_vega_position()`  
- `plot_vega_exposure()`  
- `plot_3d_options()`  

---

## ⚙️ Setup Instructions

### Prerequisites
```bash
pip install yfinance numpy pandas matplotlib scipy scikit-learn arch statsmodels tkinter
```

### Run Example
```bash
python "American Option Valuation Tool.py"
```
Each script is self-contained; run individually as a standalone module.

---

## 📊 Dependencies

| Library | Purpose |
|----------|----------|
| `yfinance` | Financial data retrieval |
| `numpy`, `pandas` | Numerical computation and data handling |
| `matplotlib` | Visualization |
| `tkinter` | GUI interface |
| `scipy`, `arch`, `statsmodels` | Optimization and statistical modeling |
| `sklearn` | Polynomial regression and ML preprocessing |

---

## 🧠 Author

**Calvin Lomax**  
Data Scientist & AI Researcher  
Focus: Quantitative modeling, portfolio optimization, and financial analytics.

---

## 🧾 License

This project is provided for **educational and personal research** purposes.  
No warranties or guarantees of financial performance are implied.
