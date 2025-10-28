import yfinance as yf
import pandas as pd

def get_options_chain(ticker):
    """
    Get options chain data for all expiry dates for the given ticker using yfinance.
    """
    stock = yf.Ticker(ticker)
    expiries = stock.options  # List of expiration dates
    options_data = []
    
    for expiry in expiries:
        calls = stock.option_chain(expiry).calls
        calls['expiry'] = expiry
        options_data.append(calls)
        
    # Combine all expiration date chains
    all_calls = pd.concat(options_data, ignore_index=True)
    return all_calls

def filter_valid_options(options_chain):
    """
    Filter options where the bid for short call exists, and the ask for long call exists.
    """
    # Exclude rows where there's no bid or ask price
    valid_options = options_chain[(options_chain['bid'] > 0) & (options_chain['ask'] > 0)]
    return valid_options

def pmcc_strategy(ticker):
    """
    Structure a Poor Man's Covered Call strategy for the input ticker.
    Ensures that the LEAPS call has a longer DTE than the short call.
    """
    # Get stock info and options chain
    stock = yf.Ticker(ticker)
    stock_price = stock.history(period='1d')['Close'].iloc[0]
    
    # Get the options chain
    options_chain = get_options_chain(ticker)
    
    # Filter for valid options (those with a bid and ask price)
    valid_options = filter_valid_options(options_chain)
    
    # Create two separate dataframes for potential LEAPS and short calls
    # The long call should be deep ITM (strike less than stock price) with a far expiry
    long_calls = valid_options[valid_options['inTheMoney'] == True]  # Deep ITM for long calls
    
    # The short call should be OTM (strike greater than stock price) with a nearer expiry
    short_calls = valid_options[valid_options['strike'] > stock_price]  # OTM for short calls
    
    if long_calls.empty or short_calls.empty:
        print("No valid combinations found.")
        return
    
    best_combination = None
    max_expected_outcome = -float('inf')  # Initialize with a large negative number
    
    # Sort both long and short call options by expiry to ensure long call has a longer expiry
    long_calls = long_calls.sort_values('expiry')
    short_calls = short_calls.sort_values('expiry')
    
    # Loop over all combinations of LEAPS and short calls where LEAPS has a longer expiry than the short call
    for _, long_call in long_calls.iterrows():
        for _, short_call in short_calls.iterrows():
            # Ensure long call expires after the short call
            if pd.to_datetime(long_call['expiry']) > pd.to_datetime(short_call['expiry']):
                # Calculate the net premium collected: (short call bid - long call ask)
                net_premium = short_call['bid'] - long_call['ask']
                
                # If this is the best combination, store it
                if net_premium > max_expected_outcome:
                    max_expected_outcome = net_premium
                    best_combination = (long_call, short_call, net_premium)
    
    if best_combination:
        long_call, short_call, expected_outcome = best_combination
        print("\nRecommended PMCC Positions:")
        print(f"LEAPS Call Option: Expiry: {long_call['expiry']}, Strike: {long_call['strike']}, Ask: {long_call['ask']}")
        print(f"Short Call Option: Expiry: {short_call['expiry']}, Strike: {short_call['strike']}, Bid: {short_call['bid']}")
        print(f"Net Position Cost: {expected_outcome}")
    else:
        print("No valid PMCC combinations found.")

# Example usage
ticker = input("Enter a stock ticker symbol: ")
pmcc_strategy(ticker)
