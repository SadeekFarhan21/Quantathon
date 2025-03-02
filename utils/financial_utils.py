"""
Utility functions for financial calculations
"""
import pandas as pd
import numpy as np

def calculate_accurate_bond_returns(df):
    """
    Calculate bond returns with proper calendar day compounding
    
    Args:
        df: DataFrame containing BondRate column (annual percentage)
        
    Returns:
        Series with properly calculated daily returns
    """
    # Create a copy to avoid modifying the original
    result = pd.Series(index=df.index, dtype='float64')
    
    # Calculate days between observations
    days_between = pd.Series(df.index).diff().dt.days.fillna(1).values
    
    # Calculate daily returns based on actual days passed
    for i in range(len(df)):
        # Get calendar days since last observation 
        days = days_between[i]
        
        # Typical case: 1 day for consecutive weekdays
        # Monday after Friday: 3 days
        # Day after holiday: variable
        
        # Convert annual rate to daily rate accounting for actual days passed
        annual_rate = df['BondRate'].iloc[i] / 100  # Convert percentage to decimal
        result.iloc[i] = (1 + annual_rate) ** (days / 365) - 1
        
    return result

def calculate_rolling_volatility(returns, window=21, annualize=True):
    """
    Calculate rolling volatility of returns
    
    Args:
        returns: Series of returns
        window: Rolling window size
        annualize: Whether to annualize volatility (√252 factor)
    
    Returns:
        Series of rolling volatility values
    """
    rolling_vol = returns.rolling(window=window, min_periods=max(5, window//4)).std()
    
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)
        
    return rolling_vol

def calculate_drawdowns(series):
    """
    Calculate drawdowns from a price or value series
    
    Args:
        series: Series of prices or portfolio values
    
    Returns:
        Series of drawdown values (as percentages)
    """
    return (series / series.cummax() - 1) * 100
