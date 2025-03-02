import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import timedelta
import importlib.util
import sys

# Check if strategy_optimizations exists and import it if available
try:
    spec = importlib.util.find_spec('config.strategy_optimizations')
    if spec is not None:
        from config.strategy_optimizations import StrategyOptimizer
        OPTIMIZATIONS_AVAILABLE = True
        optimizer = StrategyOptimizer()
        logging.info("Strategy optimizations module found and loaded")
    else:
        OPTIMIZATIONS_AVAILABLE = False
        logging.info("Strategy optimizations module not found - using default parameters")
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    logging.info("Strategy optimizations module not found - using default parameters")

# Rest of the imports and setup
from scipy import stats  # For stats.norm.cdf function

# Import utility functions
try:
    from utils.financial_utils import calculate_accurate_bond_returns
    ACCURATE_BOND_RETURNS = True
    logging.info("Using accurate bond return calculations")
except ImportError:
    ACCURATE_BOND_RETURNS = False
    logging.info("Financial utilities not found - using simplified bond return calculations")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrategyBacktester:
    def __init__(self, data, initial_capital=10000.0, use_optimized=True):
        """
        Initialize the backtester with market data and predictions
        
        Args:
            data (DataFrame): Market data with predictions
            initial_capital (float): Initial capital for backtesting
            use_optimized (bool): Whether to use optimized strategy parameters
        """
        self.df = data.copy()  # Store a copy to avoid pandas warnings
        self.initial_capital = initial_capital
        self.results = {}
        self.use_optimized = use_optimized and OPTIMIZATIONS_AVAILABLE
        self.optimizer = optimizer if OPTIMIZATIONS_AVAILABLE else None
        
        if self.use_optimized:
            logging.info("Using optimized strategy parameters")
            
        # Pre-calculate accurate bond returns if possible
        if ACCURATE_BOND_RETURNS:
            self.bond_returns = calculate_accurate_bond_returns(self.df)
        else:
            self.bond_returns = None
        
    def run_buy_and_hold(self):
        """
        Run a simple buy-and-hold strategy for comparison.
        
        Returns:
            DataFrame: Strategy performance
        """
        logging.info("Running buy-and-hold strategy backtest")
        
        # Create a deep copy to avoid SettingWithCopyWarning
        df = self.df.copy(deep=True)
        
        # Calculate daily returns for S&P 500
        df['Daily_Return'] = df['SP500'].pct_change()
        
        # Debug info to verify returns
        start_price = df['SP500'].iloc[0]
        end_price = df['SP500'].iloc[-1]
        total_market_return = (end_price / start_price - 1) * 100
        logging.info(f"S&P 500 start: {start_price:.2f}, end: {end_price:.2f}, total return: {total_market_return:.2f}%")
        
        # Calculate shares bought (based on initial capital)
        shares = self.initial_capital / start_price
        
        # Calculate portfolio value based on share price (this directly mirrors S&P 500 performance)
        df['Portfolio_Value'] = df['SP500'] * shares
        
        # Calculate cumulative returns
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        
        # Double-check final return calculation
        final_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        logging.info(f"Buy & Hold final return: {final_return:.2f}% (should match S&P 500 return: {total_market_return:.2f}%)")
        
        # Store results
        self.results['buy_and_hold'] = df[['SP500', 'BondRate', 'Daily_Return', 'Portfolio_Value', 'Cumulative_Return']]
        
        return self.results['buy_and_hold']

    def run_prediction_strategy(self, prediction_col='Predicted_Market'):
        """
        Run a strategy based on market state predictions with NO leverage
        and ensure 100% allocation between stocks and bonds.
        
        Args:
            prediction_col (str): Column name for market state predictions
            
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and self.optimizer:
            params = self.optimizer.get_parameters('prediction_strategy')
            if params:
                logging.info(f"Using optimized parameters for prediction strategy")
        
        # Standard implementation continues below
        logging.info("Running prediction-based strategy backtest")
        
        if prediction_col not in self.df.columns:
            logging.error(f"Prediction column {prediction_col} not found in market data")
            return None
            
        # Create a deep copy to avoid SettingWithCopyWarning
        df = self.df.copy(deep=True)
        
        # Calculate returns and volatility
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        df['Rolling_Vol_21d'] = df['SP500_Return'].rolling(window=21).std() * np.sqrt(252)
        df['Rolling_Vol_63d'] = df['SP500_Return'].rolling(window=63).std() * np.sqrt(252)
        
        # Calculate trend strength indicators
        df['MA_45'] = df['SP500'].rolling(window=45).mean()  # Optimized from 50
        df['MA_180'] = df['SP500'].rolling(window=180).mean()  # Optimized from 200
        df['Trend_Strength'] = (df['SP500'] / df['MA_180'] - 1) * 100
        
        # Correct bond rate calculation for short-term interest rates (annual to daily)
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Define the base allocation based on market prediction
        bull_alloc = params.get('bull_allocation', 1.0)
        static_alloc = params.get('static_allocation', 0.6)
        bear_alloc = params.get('bear_allocation', 0.0)
        
        df['Base_Allocation'] = np.where(df[prediction_col] == 'Bull', bull_alloc,
                                 np.where(df[prediction_col] == 'Static', static_alloc, bear_alloc))
        
        # Define conditions for allocation adjustments
        trend_strength_threshold = params.get('trend_strength_threshold', 4.0)
        vol_threshold_high = params.get('vol_threshold_high', 0.22)
        vol_spike_threshold = params.get('vol_spike_threshold', 1.4)
        
        strong_bull = (df['Trend_Strength'] > trend_strength_threshold) & (df['Rolling_Vol_63d'] < 0.15) & (df[prediction_col] == 'Bull')
        high_vol = df['Rolling_Vol_21d'] > vol_threshold_high
        vol_spike = df['Rolling_Vol_21d'] > df['Rolling_Vol_63d'] * vol_spike_threshold
        
        # Apply allocation adjustments
        df['SP500_Allocation'] = df['Base_Allocation'].copy()
        df.loc[strong_bull, 'SP500_Allocation'] = 1.0  # Max 100% allocation
        
        vol_dampening = params.get('vol_dampening', 0.6)
        vol_spike_dampening = params.get('vol_spike_dampening', 0.65)
        
        # Reduce exposure in high volatility environments
        df.loc[high_vol, 'SP500_Allocation'] *= vol_dampening
        df.loc[vol_spike, 'SP500_Allocation'] *= vol_spike_dampening
        
        # Ensure allocations stay within allowed range (0-100%)
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)
        
        # ALWAYS ensure 100% allocation - bonds get whatever is not in stocks
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Verify 100% allocation
        total_allocation = df['SP500_Allocation'] + df['Bond_Allocation']
        if not np.allclose(total_allocation, 1.0, atol=1e-10):
            logging.warning("Found allocations not summing to 100%. Fixing...")
            df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Log allocation statistics
        full_equity_days = (df['SP500_Allocation'] >= 0.99).sum()
        full_bonds_days = (df['SP500_Allocation'] <= 0.01).sum()
        balanced_days = ((df['SP500_Allocation'] > 0.4) & (df['SP500_Allocation'] < 0.6)).sum()
        logging.info(f"Allocation statistics: {full_equity_days} days with ≥99% equity, " +
                   f"{full_bonds_days} days with ≤1% equity exposure, {balanced_days} days balanced")
        
        # Initialize portfolio tracking
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
        
        # Calculate portfolio returns
        for i in range(1, len(df)):
            # Use previous day's allocation but today's returns (realistic implementation)
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            # Apply portfolio return calculation
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return and drawdown
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Store results
        self.results['prediction_strategy'] = df[['SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return', 
                         'Predicted_Market', 'SP500_Allocation', 'Bond_Allocation',
                         'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown']]
        
        return self.results['prediction_strategy']

    def run_dynamic_allocation_strategy(self):
        """
        Run a dynamic allocation strategy that uses prediction probabilities
        with 100% capital allocation at all times.
        
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and self.optimizer:
            params = self.optimizer.get_parameters('dynamic_allocation')
            if params:
                logging.info(f"Using optimized parameters for dynamic allocation strategy")
        
        logging.info("Running dynamic allocation strategy")
        
        # Create working copy of data
        df = self.df.copy(deep=True)
        
        # Check if we have probability columns
        has_probs = all(col in df.columns for col in ['Bull_Prob', 'Bear_Prob'])
        
        if not has_probs:
            logging.warning("Probability columns not found. Creating dummy probabilities from predictions.")
            if 'Predicted_Market' not in df.columns:
                logging.warning("No predictions found! Using actual market state.")
                df['Predicted_Market'] = df['Market_State'] if 'Market_State' in df.columns else 'Static'
                
            # Create dummy probability columns
            df['Bull_Prob'] = 0.0
            df['Bear_Prob'] = 0.0
            df['Static_Prob'] = 0.0
            
            # Assign probabilities based on predicted state
            df.loc[df['Predicted_Market'] == 'Bull', 'Bull_Prob'] = 0.7
            df.loc[df['Predicted_Market'] == 'Bull', 'Static_Prob'] = 0.2
            df.loc[df['Predicted_Market'] == 'Bull', 'Bear_Prob'] = 0.1
            
            df.loc[df['Predicted_Market'] == 'Static', 'Static_Prob'] = 0.7
            df.loc[df['Predicted_Market'] == 'Static', 'Bull_Prob'] = 0.15
            df.loc[df['Predicted_Market'] == 'Static', 'Bear_Prob'] = 0.15
            
            df.loc[df['Predicted_Market'] == 'Bear', 'Bear_Prob'] = 0.7
            df.loc[df['Predicted_Market'] == 'Bear', 'Static_Prob'] = 0.2
            df.loc[df['Predicted_Market'] == 'Bear', 'Bull_Prob'] = 0.1
        
        # Calculate returns for securities
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)  # Fill first day with 0
        
        # Use the accurate bond return calculation if available, otherwise use the simple method
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Calculate rolling volatility for risk adjustment
        df['Rolling_Vol'] = df['SP500_Return'].rolling(window=21, min_periods=1).std() * np.sqrt(252)
        df['Rolling_Vol'] = df['Rolling_Vol'].fillna(df['Rolling_Vol'].mean())
        
        # Define allocation based on probabilities - ensure 0 to 100% range
        bull_prob_weight = params.get('bull_prob_weight', 1.2)
        bear_prob_weight = params.get('bear_prob_weight', 1.0)
        
        df['SP500_Allocation'] = (bull_prob_weight * df['Bull_Prob'] - bear_prob_weight * df['Bear_Prob'] + 0.5).clip(0, 1.0)
        
        # Apply volatility adjustment - reduce equity allocation in high vol environments
        high_vol_threshold = params.get('high_vol_threshold', 0.25)
        vol_adjustment = params.get('vol_adjustment', 0.8)
        
        high_vol = df['Rolling_Vol'] > high_vol_threshold
        df.loc[high_vol, 'SP500_Allocation'] *= vol_adjustment
        
        # Ensure valid range
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)
        
        # ALWAYS ensure 100% allocation - bonds get whatever is not in stocks
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Double-verify 100% allocation
        total_allocation = df['SP500_Allocation'] + df['Bond_Allocation']
        if not np.allclose(total_allocation, 1.0, atol=1e-10):
            logging.error(f"Dynamic allocation error: Min={total_allocation.min()}, Max={total_allocation.max()}")
            # Fix by recalculating bond allocation
            df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Calculate portfolio returns
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
        
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return and drawdowns
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Log allocation distribution
        fully_in_stocks = (df['SP500_Allocation'] > 0.95).sum()
        fully_in_bonds = (df['SP500_Allocation'] < 0.05).sum()
        balanced = ((df['SP500_Allocation'] >= 0.45) & (df['SP500_Allocation'] <= 0.55)).sum()
        logging.info(f"Dynamic allocation: {fully_in_stocks} days >95% stocks, {fully_in_bonds} days >95% bonds, {balanced} days ~50/50 split") 
        
        # Store results - include Predicted_Market if available
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'SP500_Allocation', 'Bond_Allocation',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        if 'Predicted_Market' in df.columns:
            result_columns.insert(4, 'Predicted_Market')
            
        self.results['dynamic_allocation'] = df[result_columns]
        
        return self.results['dynamic_allocation']

    def run_anomaly_aware_strategy(self, anomaly_col='ensemble_anomaly', prediction_col='Predicted_Market'):
        """
        Run a strategy that adapts to detected market anomalies,
        ensuring 100% capital allocation at all times (stocks + bonds = 100%)
        
        Args:
            anomaly_col (str): Column with anomaly flags (-1 for anomalies, 1 for normal)
            prediction_col (str): Column with market predictions
            
        Returns:
            DataFrame: Strategy performance results
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and self.optimizer:
            params = self.optimizer.get_parameters('anomaly_aware')
            if params:
                logging.info(f"Using optimized parameters for anomaly aware strategy")
        
        # Standard implementation continues here
        logging.info("Running anomaly-aware investment strategy")
        
        if anomaly_col not in self.df.columns:
            logging.warning(f"Anomaly column {anomaly_col} not found. Falling back to standard strategy.")
            return self.run_prediction_strategy(prediction_col)
        
        # Create a deep copy to avoid SettingWithCopyWarning
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Correct bond rate calculation - convert from annual percentage to daily decimal
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Base allocation on predictions
        df['Base_Allocation'] = np.where(df[prediction_col] == 'Bull', 1.0,
                                   np.where(df[prediction_col] == 'Static', 0.5, 0.0))
        
        # Initialize allocation columns
        df['SP500_Allocation'] = df['Base_Allocation'].copy()
        df['Anomaly_Adjustment'] = 0.0
        
        # Track days since anomaly
        df['Days_Since_Anomaly'] = 9999  # Large default value
        
        # Process anomalies and adjust allocations
        anomaly_dates = df[df[anomaly_col] == -1].index
        recovery_period = params.get('recovery_period', 10)  # Days to gradually restore allocation after anomaly
        
        # Calculate days since most recent anomaly
        if len(anomaly_dates) > 0:
            for i, date in enumerate(df.index):
                # Find closest anomaly before this date
                prev_anomalies = [a for a in anomaly_dates if a <= date]
                if prev_anomalies:
                    most_recent = max(prev_anomalies)
                    days_since = (date - most_recent).days
                    df.loc[date, 'Days_Since_Anomaly'] = days_since
        
        # Adjust allocations based on days since anomaly
        for idx, row in df.iterrows():
            days_since = row['Days_Since_Anomaly']
            if days_since == 0:  # Day of anomaly
                # Complete risk-off response - set stock allocation to 0 (100% bonds)
                df.loc[idx, 'SP500_Allocation'] = 0.0
            elif days_since < recovery_period:
                # Gradually restore allocation
                recovery_factor = days_since / recovery_period
                df.loc[idx, 'SP500_Allocation'] = row['Base_Allocation'] * recovery_factor
        
        # Ensure allocation is in valid range
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)
        
        # ALWAYS ensure 100% allocation - bonds get whatever is not in stocks
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Verify 100% allocation
        total_allocation = df['SP500_Allocation'] + df['Bond_Allocation']
        if not np.allclose(total_allocation, 1.0, atol=1e-10):
            logging.error(f"Anomaly strategy allocation error: Min={total_allocation.min()}, Max={total_allocation.max()}")
            # Fix by recalculating bond allocation
            df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Calculate portfolio returns
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital  # Set first value
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
        
        for i in range(1, len(df)):
            # Use previous day's allocation
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return and drawdown
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Log anomaly impact statistics
        anomaly_count = (df['Days_Since_Anomaly'] == 0).sum()
        zero_alloc_days = (df['SP500_Allocation'] == 0).sum()
        logging.info(f"Anomaly strategy: {anomaly_count} anomalies detected, {zero_alloc_days} days with 100% bond allocation")
        
        # Store results with strategy name 'anomaly_aware'
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'SP500_Allocation', 'Bond_Allocation',
            'Days_Since_Anomaly', 'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        self.results['anomaly_aware'] = df[result_columns]
        
        return self.results['anomaly_aware']

    def _identify_drawdown_periods(self, df):
        """
        Identify drawdown periods based on Portfolio_Value.
        
        Args:
            df (DataFrame): DataFrame containing 'Portfolio_Value'
            
        Returns:
            list of tuples: Each tuple is (start_date, end_date, min_drawdown)
        """
        if 'Portfolio_Value' not in df.columns:
            return None
        
        cum_returns = df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1
        periods = []
        in_drawdown = False
        start = None
        
        for date, value in cum_returns.items():  # items() is the correct method to use
            if not in_drawdown and value < 0:
                in_drawdown = True
                start = date
            elif in_drawdown and value == 0:
                end = date
                periods.append((start, end, cum_returns.loc[start:end].min()))
                in_drawdown = False
        
        if in_drawdown:
            end = df.index[-1]
            periods.append((start, end, cum_returns.loc[start:end].min()))
        
        return periods

    def run_combined_strategy(self):
        """
        Run a combined strategy that integrates multiple indicators with risk management,
        ensuring 100% capital allocation at all times.
        
        Returns:
            DataFrame: Strategy performance metrics
        """
        # If optimizations are available and enabled, use optimized version
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            try:
                # Apply optimizations to self
                self = optimizer._optimize_combined_strategy(self)
                # Call the method again with optimization disabled to avoid infinite recursion
                saved_setting = self.use_optimized
                self.use_optimized = False
                result = self.run_combined_strategy()
                self.use_optimized = saved_setting
                return result
            except Exception as e:
                logging.error(f"Error in optimized combined strategy: {str(e)}. Using standard version.")
        
        # Standard implementation continues here
        logging.info("Running combined investment strategy")
        
        try:
            # Create working copy of data
            df = self.df.copy(deep=True)
            
            # 1. PREPARE DATA
            # Return calculations
            df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
            
            # Use the accurate bond return calculation if available, otherwise use the simple method
            if self.bond_returns is not None:
                df['Daily_Bond_Return'] = self.bond_returns
            else:
                df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
                
            # 2. CALCULATE INDICATORS
            # Prediction-based signal
            if 'Predicted_Market' in df.columns:
                df['Pred_Signal'] = np.where(df['Predicted_Market'] == 'Bull', 0.8,
                                    np.where(df['Predicted_Market'] == 'Static', 0.5, 0.2))
                                    
            elif all(col in df.columns for col in ['Bull_Prob', 'Bear_Prob']):
                df['Pred_Signal'] = (df['Bull_Prob'] - df['Bear_Prob'] + 0.5).clip(0, 1)
            else:
                df['Pred_Signal'] = 0.5  # Default neutral
                logging.warning("No prediction data found! Using neutral prediction signal.")
            
            # Calculate technical indicators
            for window in [20, 50, 100, 200]:
                df[f'MA_{window}'] = df['SP500'].rolling(window=window, min_periods=max(5, window//10)).mean()
                
            # Fill any remaining NaN values with the first valid value
            for col in df.columns:
                if (col.startswith('MA_')):
                    df[col] = df[col].fillna(method='bfill').fillna(df['SP500'].iloc[0])
                    
            # Calculate price-to-MA ratio
            for window in [20, 50, 100, 200]:
                df[f'Price_vs_MA{window}'] = (df['SP500'] / df[f'MA_{window}']) - 1
            
            # Combined trend signal with error handling
            df['Trend_Signal'] = 0.5  # Default neutral value
            
            # Only calculate if we have all required MAs
            required_cols = ['SP500', 'MA_20', 'MA_50', 'MA_100', 'MA_200']
            if all(col in df.columns for col in required_cols):
                df['Trend_Signal'] = (
                    0.4 * (df['SP500'] > df['MA_20']).astype(float) +
                    0.3 * (df['SP500'] > df['MA_50']).astype(float) +
                    0.2 * (df['SP500'] > df['MA_100']).astype(float) +
                    0.1 * (df['SP500'] > df['MA_200']).astype(float)
                )
            
            # 3. VOLATILITY INDICATORS - with proper error handling
            df['Vol_21d'] = df['SP500_Return'].rolling(window=21, min_periods=5).std() * np.sqrt(252)
            df['Vol_63d'] = df['SP500_Return'].rolling(window=63, min_periods=21).std() * np.sqrt(252)
            
            # Fill NaN values with reasonable defaults
            default_vol = 0.15  # 15% annualized vol as reasonable default
            df['Vol_21d'] = df['Vol_21d'].fillna(default_vol)
            df['Vol_63d'] = df['Vol_63d'].fillna(default_vol)
            
            # Normalize volatility (lower is better for allocation) - cap at 0-1 range
            df['Vol_Signal'] = 1 - (df['Vol_21d'] / 0.20).clip(0, 1)
            
            # Detect volatility regime with safety checks
            safe_min_value = 0.001  # To avoid division by zero
            df['Vol_Ratio'] = df['Vol_21d'] / df['Vol_63d'].replace(0, safe_min_value)
            df['Vol_Regime'] = np.where(df['Vol_Ratio'] > 1.2, 'Expanding', 
                               np.where(df['Vol_Ratio'] < 0.8, 'Contracting', 'Normal'))
            
            # 4. COMBINE ALL SIGNALS - weighted approach with validation
            # Set signal weights (must sum to 1.0)
            risk_weight = 0.5    # Prediction signal weight
            trend_weight = 0.3   # Trend signal weight
            vol_weight = 0.2     # Volatility signal weight
            
            # Ensure weights sum to 1.0
            total_weight = risk_weight + trend_weight + vol_weight
            if abs(total_weight - 1.0) > 0.001:
                risk_weight = risk_weight / total_weight
                trend_weight = trend_weight / total_weight
                vol_weight = vol_weight / total_weight
                
            df['Combined_Signal'] = (
                risk_weight * df['Pred_Signal'].fillna(0.5) +
                trend_weight * df['Trend_Signal'].fillna(0.5) +
                vol_weight * df['Vol_Signal'].fillna(0.5)
            )
            
            # 5. DYNAMIC POSITION SIZING with safety checks
            # Base position - ensure it's in valid range (0-100%)
            df['Base_Position'] = df['Combined_Signal'].clip(0, 1)
            
            # Calculate additional factors with safe computation
            # Add momentum calculation with proper handling of NaNs
            df['Momentum_3m'] = df['SP500'].pct_change(63).fillna(0)
            df['Trend_Strength'] = df['Momentum_3m'].clip(-0.2, 0.2) / 0.2  # Normalize to [-1,1]
            
            # Apply trend strength multiplier (no leverage allowed)
            df['Trend_Multiplier'] = 1.0 + 0.2 * df['Trend_Strength'].clip(0, 1)  # Max 1.2 multiplier
            
            # Volatility adjustment - reduce position in high volatility
            df['Vol_Adjuster'] = np.where(df['Vol_Regime'] == 'Expanding', 0.7,  # Reduce by 30% in expanding vol
                                  np.where(df['Vol_Regime'] == 'Contracting', 1.1, 1.0))  # Increase by 10% in contracting vol
            
            # Calculate final allocation with all factors - ensure valid values at every step
            df['SP500_Allocation'] = (
                df['Base_Position'].fillna(0.5) * 
                df['Trend_Multiplier'].fillna(1.0) * 
                df['Vol_Adjuster'].fillna(1.0)
            ).clip(0, 1.0)  # Hard cap at 100% to avoid leverage
            
            # 6. TACTICAL ADJUSTMENTS
            # Add stop-loss logic: reduce equity exposure after significant drawdowns
            df['Rolling_Return_5d'] = df['SP500'].pct_change(5).fillna(0)
            significant_drop = df['Rolling_Return_5d'] < -0.07  # 7% drop in 5 days
            
            # Apply stop-loss and time-based recovery with careful indexing
            for i in range(1, len(df)):
                if significant_drop.iloc[i]:
                    # Only apply if we have a valid allocation
                    if not np.isnan(df['SP500_Allocation'].iloc[i]):
                        # Apply stop-loss: reduce position by 50%
                        df.loc[df.index[i], 'SP500_Allocation'] = df.loc[df.index[i], 'SP500_Allocation'] * 0.5
                        
                        # Gradually restore position over next 10 days
                        recovery_days = min(10, len(df) - i - 1)
                        for j in range(1, recovery_days + 1):
                            if i + j < len(df):
                                recovery_factor = j / recovery_days  # Linear recovery
                                # Use explicit .loc indexing to avoid SettingWithCopyWarning
                                df.loc[df.index[i+j], 'SP500_Allocation'] = min(
                                    df.loc[df.index[i+j], 'SP500_Allocation'],  # Keep minimum of calculated or recovered
                                    df.loc[df.index[i], 'SP500_Allocation'] * (1 + recovery_factor * 0.5 * 2)  # Restore the 50% reduction
                                )
            
            # 7. FINALIZE ALLOCATIONS - ensure valid values
            df['SP500_Allocation'] = df['SP500_Allocation'].fillna(0.5)  # Default to 50/50 if NaN
            df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)  # Ensure no leverage
            df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
            
            # 8. CALCULATE PORTFOLIO RETURNS with careful handling of edge cases
            df['Portfolio_Return'] = np.nan
            df['Portfolio_Value'] = np.nan
            df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
            df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
            
            for i in range(1, len(df)):
                try:
                    sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
                    bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
                    
                    portfolio_return = sp500_contrib + bond_contrib
                    
                    # Safety check for unreasonable returns
                    if portfolio_return < -0.5 or portfolio_return > 0.5:
                        logging.warning(f"Suspicious portfolio return on {df.index[i]}: {portfolio_return:.4f}")
                        portfolio_return = np.clip(portfolio_return, -0.2, 0.2)  # Reasonable bounds
                        
                    df.loc[df.index[i], 'Portfolio_Return'] = portfolio_return
                    df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + portfolio_return)
                    
                    # Safety check for negative portfolio value (shouldn't happen with normal returns)
                    if df.loc[df.index[i], 'Portfolio_Value'] <= 0:
                        logging.error(f"Negative portfolio value detected on {df.index[i]}")
                        df.loc[df.index[i], 'Portfolio_Value'] = self.initial_capital * 0.1  # Prevent total loss
                except Exception as e:
                    logging.error(f"Error calculating portfolio return for day {i}: {str(e)}")
                    # Use previous values or safe defaults
                    df.loc[df.index[i], 'Portfolio_Return'] = 0.0
                    df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] if i > 0 else self.initial_capital
            
            # 9. CALCULATE PERFORMANCE METRICS with NaN handling
            df['Cumulative_Return'] = (df['Portfolio_Value'] / self.initial_capital - 1).fillna(0)
            df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).fillna(0) * 100
            
            # Store results with only essential columns to reduce space
            self.results['combined_strategy'] = df[[
                'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
                'Pred_Signal', 'Trend_Signal', 'Vol_Signal', 'Combined_Signal',
                'SP500_Allocation', 'Bond_Allocation',
                'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
            ]]
            
            # Log final performance
            final_return = df['Cumulative_Return'].iloc[-1] * 100 if len(df) > 0 else 0
            max_dd = df['Drawdown'].min() if len(df) > 0 else 0
            logging.info(f"Combined strategy - Final return: {final_return:.2f}%, Max drawdown: {max_dd:.2f}%")
            
            return self.results['combined_strategy']
            
        except Exception as e:
            logging.error(f"Error in combined strategy: {str(e)}")
            # Create a minimal valid dataframe to return
            fallback_df = pd.DataFrame(index=self.df.index)
            fallback_df['SP500'] = self.df['SP500']
            fallback_df['BondRate'] = self.df['BondRate']
            fallback_df['SP500_Return'] = self.df['SP500'].pct_change()
            fallback_df['Daily_Bond_Return'] = (1 + self.df['BondRate'] / 100) ** (1/252) - 1
            fallback_df['SP500_Allocation'] = 0.5
            fallback_df['Bond_Allocation'] = 0.5
            fallback_df['Portfolio_Return'] = 0.0
            fallback_df['Portfolio_Value'] = self.initial_capital
            fallback_df['Cumulative_Return'] = 0.0
            fallback_df['Drawdown'] = 0.0
            
            self.results['combined_strategy'] = fallback_df[[
                'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
                'SP500_Allocation', 'Bond_Allocation',
                'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
            ]]
            
            return self.results['combined_strategy']

    def calculate_metrics(self, strategy_name='prediction_strategy'):
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            dict: Performance metrics
        """
        if strategy_name not in self.results:
            logging.error(f"Strategy {strategy_name} not found in results")
            return {}
            
        df = self.results[strategy_name]
        
        # Find correct column names for portfolio value and returns
        if 'Portfolio_Value' in df.columns:
            portfolio_value_col = 'Portfolio_Value'
        elif f'Portfolio_Value_{strategy_name}' in df.columns:
            portfolio_value_col = f'Portfolio_Value_{strategy_name}'
        else:
            logging.error(f"Portfolio value column not found for strategy {strategy_name}")
            return {}
        
        if 'Portfolio_Return' in df.columns:
            return_col = 'Portfolio_Return'
        elif f'Portfolio_Return_{strategy_name}' in df.columns:
            return_col = f'Portfolio_Return_{strategy_name}'
        elif 'Daily_Return' in df.columns:
            return_col = 'Daily_Return'
        else:
            logging.error(f"Return column not found for strategy {strategy_name}")
            return {}
        
        # Calculate metrics
        returns = df[return_col].dropna()
        
        metrics = {
            'Total_Return': (df[portfolio_value_col].iloc[-1] / self.initial_capital - 1) * 100,
            'Annualized_Return': (df[portfolio_value_col].iloc[-1] / self.initial_capital) ** (252 / len(df)) - 1,
            'Daily_Volatility': returns.std() * 100,
            'Annualized_Volatility': returns.std() * np.sqrt(252) * 100,
            'Sharpe_Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'Max_Drawdown': (df[portfolio_value_col] / df[portfolio_value_col].cummax() - 1).min() * 100,
            'Win_Rate': (returns > 0).sum() / len(returns) * 100,
        }
        
        return metrics

    def summary(self, include_risk_metrics=True):
        """
        Generate performance summary for all strategies with enhanced risk metrics
        
        Args:
            include_risk_metrics (bool): Whether to include comprehensive risk metrics
            
        Returns:
            DataFrame: Summary metrics for all strategies
        """
        summary_data = {}
        additional_metrics = {}
        
        for strategy_name in self.results.keys():
            metrics = self.calculate_metrics(strategy_name)
            summary_data[strategy_name] = metrics
            
            # Initialize a dictionary to store additional metrics for this strategy
            additional_metrics[strategy_name] = {}
        
        summary_df = pd.DataFrame(summary_data).T
        
        # Print summary
        logging.info("Strategy Performance Summary:")
        # Format and print in a clean way
        for strategy, metrics in summary_df.iterrows():
            logging.info(f"\n{'='*50}")
            logging.info(f"Strategy: {strategy.replace('_', ' ').title()}")
            logging.info(f"{'='*50}")
            logging.info(f"Total Return: {metrics['Total_Return']:.2f}%")
            logging.info(f"Annualized Return: {metrics['Annualized_Return']*100:.2f}%")
            logging.info(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
            logging.info(f"Max Drawdown: {metrics['Max_Drawdown']:.2f}%")
            logging.info(f"Win Rate: {metrics['Win_Rate']:.2f}%")
        
        # Add advanced risk metrics if requested
        if include_risk_metrics:
            # Add volatility metrics
            for strategy, result_df in self.results.items():
                # Average volatility over different periods
                for period in [21, 63]:
                    vol_col = f'Rolling_Vol_{period}d'
                    if vol_col in result_df.columns:
                        additional_metrics[strategy][f'Volatility_{period}d'] = result_df[vol_col].mean()
                # Downside risk measures
                if 'Downside_Vol_63d' in result_df.columns:
                    additional_metrics[strategy]['Downside_Volatility'] = result_df['Downside_Vol_63d'].mean()
                # VaR and CVaR
                if 'Portfolio_Return' in result_df.columns:
                    returns = result_df['Portfolio_Return'].dropna()
                elif f'Portfolio_Return_{strategy}' in result_df.columns:
                    returns = result_df[f'Portfolio_Return_{strategy}'].dropna()
                else:
                    returns = result_df['Daily_Return'].dropna() if 'Daily_Return' in result_df.columns else pd.Series()
                if len(returns) > 0:
                    additional_metrics[strategy]['VaR_95'] = np.percentile(returns, 5)
                    additional_metrics[strategy]['VaR_99'] = np.percentile(returns, 1)
                    additional_metrics[strategy]['CVaR_95'] = returns[returns <= additional_metrics[strategy]['VaR_95']].mean()
                
                # Worst day/week/month
                if len(returns) >= 5:  # Ensure we have enough data
                    additional_metrics[strategy]['Worst_Day'] = returns.min()
                    # Calculate worst week if we have enough data
                    if len(returns) >= 5:
                        weekly_returns = (1 + returns).resample('W').prod() - 1
                        additional_metrics[strategy]['Worst_Week'] = weekly_returns.min()
                    # Calculate worst month if we have enough data
                    if len(returns) >= 21:
                        monthly_returns = (1 + returns).resample('M').prod() - 1
                        additional_metrics[strategy]['Worst_Month'] = monthly_returns.min()
                
                # Sortino Ratio
                if 'Downside_Vol_63d' in result_df.columns and result_df['Downside_Vol_63d'].mean() > 0:
                    additional_metrics[strategy]['Sortino_Ratio'] = (
                        summary_df.loc[strategy, 'Annualized_Return'] / 
                        result_df['Downside_Vol_63d'].mean()
                    )
                # Calmar Ratio
                if 'Calmar_Ratio' in result_df.columns:
                    additional_metrics[strategy]['Calmar_Ratio'] = result_df['Calmar_Ratio'].mean()
                # Recovery stats
                drawdown_periods = self._identify_drawdown_periods(result_df)
                if drawdown_periods:
                    recovery_times = [end - start for start, end, _ in drawdown_periods if end is not None]
                    if recovery_times:
                        additional_metrics[strategy]['Avg_Recovery_Days'] = np.mean([rt.days for rt in recovery_times])
                        additional_metrics[strategy]['Max_Recovery_Days'] = max([rt.days for rt in recovery_times])
            
            # Convert additional metrics to DataFrame and combine with summary_df
            additional_metrics_df = pd.DataFrame(additional_metrics).T
            summary_df = pd.concat([summary_df, additional_metrics_df], axis=1)
        
        return summary_df

    def plot_performance(self, save_path='results/strategy_performance.png'):
        """
        Plot performance comparison of strategies.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(14, 8))
        
        # Plot each strategy's cumulative return
        for strategy_name, df in self.results.items():
            # Check if Cumulative_Return exists, otherwise calculate it
            if 'Cumulative_Return' not in df.columns:
                if 'Portfolio_Value' in df.columns:
                    cumulative_return = df['Portfolio_Value'] / self.initial_capital - 1
                elif f'Portfolio_Value_{strategy_name}' in df.columns:
                    cumulative_return = df[f'Portfolio_Value_{strategy_name}'] / self.initial_capital - 1
                else:
                    logging.warning(f"Cannot calculate cumulative return for {strategy_name}. No portfolio value column found.")
                    continue
            else:
                cumulative_return = df['Cumulative_Return']
            
            # Plot cumulative return
            plt.plot(df.index, cumulative_return, label=f"{strategy_name.replace('_', ' ').title()}")
            
            # Add annotation for final return
            final_return = cumulative_return.iloc[-1] * 100
            plt.annotate(
                f"{final_return:.1f}%", 
                xy=(df.index[-1], cumulative_return.iloc[-1]),
                xytext=(10, 0),
                textcoords='offset points',
                va='center'
            )
        
        plt.title('Strategy Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Performance comparison plot saved to {save_path}")

    def plot_allocations(self, strategy_name=None, save_path="allocations.png"):
        """
        Plot allocations over time for a specific strategy
        
        Args:
            strategy_name (str): Name of strategy to plot allocations for
            save_path (str): Path to save the plot
        
        Returns:
            None
        """
        # Retrieve the DataFrame corresponding to the strategy
        if strategy_name:
            if strategy_name in self.results:
                df = self.results[strategy_name]
            else:
                logging.warning(f"Strategy '{strategy_name}' not found in results; using first available strategy.")
                df = next(iter(self.results.values()))
        else:
            # Use the first available strategy if no name provided
            df = next(iter(self.results.values()))
        
        # Determine allocation column; here we expect 'SP500_Allocation'
        if 'SP500_Allocation' in df.columns:
            allocation_col = 'SP500_Allocation'
        else:
            allocation_col = None
        
        if allocation_col:
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df[allocation_col])
            plt.xlabel("Date")
            plt.ylabel("Allocation")
            plt.title(f"Allocation Over Time ({strategy_name if strategy_name else 'Default'})")
        else:
            plt.figure()
            plt.text(0.5, 0.5, "No allocation data available", ha="center")
            plt.title("Allocation Data")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Allocation plot saved to {save_path}")

    def _verify_allocation_totals(self, df, equity_col='SP500_Allocation', bond_col='Bond_Allocation'):
        """
        Verify and fix allocations to ensure they sum to exactly 100%
        
        Args:
            df (DataFrame): DataFrame with allocations
            equity_col (str): Name of equity allocation column
            bond_col (str): Name of bond allocation column
            
        Returns:
            DataFrame: DataFrame with verified allocations
        """
        # First ensure equity allocation is within valid range (0-100%)
        df[equity_col] = df[equity_col].clip(0, 1.0)
        
        # Always calculate bond allocation as complement to ensure 100% total
        df[bond_col] = 1.0 - df[equity_col]
        
        # Double-check the totals
        total_allocation = df[equity_col] + df[bond_col]
        
        # Check for NaN or Inf values before normalization
        if not np.all(np.isfinite(total_allocation)):
            logging.warning("Non-finite values detected in total allocation before normalization. Cleaning...")
            df[[equity_col, bond_col]] = df[[equity_col, bond_col]].fillna(0)  # Replace NaN/Inf with 0
            total_allocation = df[equity_col] + df[bond_col]
        
        # Use a more robust check for near-equality
        if not np.allclose(total_allocation, 1.0, rtol=1e-05, atol=1e-08):
            logging.warning(f"Found allocations not summing to 100%. Min: {total_allocation.min()}, Max: {total_allocation.max()}")
            
            # Force fix by recalculating bond allocation
            df[bond_col] = 1.0 - df[equity_col]
            
            # Final verification
            new_total = df[equity_col] + df[bond_col]
            
            # Check for NaN or Inf values after recalculating bond allocation
            if not np.all(np.isfinite(new_total)):
                logging.warning("Non-finite values detected in total allocation after recalculation. Cleaning...")
                df[[equity_col, bond_col]] = df[[equity_col, bond_col]].fillna(0)  # Replace NaN/Inf with 0
                new_total = df[equity_col] + df[bond_col]
            
            if not np.allclose(new_total, 1.0, rtol=1e-05, atol=1e-08):
                logging.error(f"CRITICAL: Failed to fix allocation totals. Check data types and values.")
                
                # As a last resort, normalize the allocations
                # Add a small amount of regularization to prevent division by zero
                regularization = 1e-9
                df[[equity_col, bond_col]] = df[[equity_col, bond_col]].div(new_total + regularization, axis=0)
                
                # Final check
                final_total = df[equity_col] + df[bond_col]
                if not np.allclose(final_total, 1.0, rtol=1e-05, atol=1e-08):
                    logging.critical("Failed to normalize allocations. Check for NaN or Inf values.")
        
        return df