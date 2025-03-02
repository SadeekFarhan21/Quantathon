import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import os
import importlib.util
from src.backtest import StrategyBacktester

# Check if strategy_optimizations exists and import it if available
try:
    spec = importlib.util.find_spec('config.strategy_optimizations')
    if spec is not None:
        from config.strategy_optimizations import StrategyOptimizer
        OPTIMIZATIONS_AVAILABLE = True
        optimizer = StrategyOptimizer()
        logging.info("Strategy optimizations module found and loaded in enhanced backtester")
    else:
        OPTIMIZATIONS_AVAILABLE = False
        logging.info("Strategy optimizations module not found - using default parameters in enhanced backtester")
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    logging.info("Strategy optimizations module not found - using default parameters in enhanced backtester")

# Try to import utility functions specifically for this class
try:
    from utils.financial_utils import calculate_accurate_bond_returns, calculate_rolling_volatility
    FINANCIAL_UTILS_AVAILABLE = True
except ImportError:
    FINANCIAL_UTILS_AVAILABLE = False

class EnhancedBacktester(StrategyBacktester):
    """
    Enhanced backtester with advanced strategies and risk management,
    ensuring 100% capital allocation at all times (stocks + bonds = 100%)
    """
    
    def __init__(self, data, initial_capital=10000.0, use_optimized=True):
        """
        Initialize enhanced backtester
        
        Args:
            data (DataFrame): Market data with predictions
            initial_capital (float): Initial capital
            use_optimized (bool): Whether to use optimized parameters
        """
        super().__init__(data, initial_capital, use_optimized)
        self.output_dir = "results"
        logging.info("Enhanced backtester initialized with 100% capital allocation policy (no leverage)")
        
        # Additional initialization for enhanced features
        if FINANCIAL_UTILS_AVAILABLE:
            # Calculate and store commonly used rolling volatilities for efficiency
            returns = self.df['SP500'].pct_change().fillna(0)
            self.vol_21d = calculate_rolling_volatility(returns, window=21)
            self.vol_63d = calculate_rolling_volatility(returns, window=63)
            self.vol_126d = calculate_rolling_volatility(returns, window=126)
    
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
    
    def run_tactical_risk_managed_strategy(self, target_vol=0.12, max_leverage=1.0):
        """
        Run a tactical risk-managed strategy with target volatility
        and ensure 100% allocation between stocks and bonds.
        
        Args:
            target_vol (float): Target annualized volatility (decimal)
            max_leverage (float): Maximum leverage allowed (capped at 1.0)
            
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('tactical_risk_managed')
            if params:
                logging.info(f"Using optimized parameters for tactical risk-managed strategy")
                # Use provided target_vol if specified, otherwise use optimized value
                if target_vol == 0.12:  # Default value
                    target_vol = params.get('target_vol', target_vol)
                logging.info(f"Target volatility: {target_vol:.2f}")
        
        # Force max_leverage to be at most 1.0 to ensure no leverage
        max_leverage = min(max_leverage, 1.0)
        if max_leverage < 1.0:
            logging.info(f"Restricting max leverage to {max_leverage:.2f}")
        
        # Create working copy of data
        df = self.df.copy(deep=True)
        
        # Calculate returns and volatility
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use pre-calculated bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Calculate rolling volatility with optimized lookback
        vol_lookback = params.get('vol_lookback', 63)  # Default to 63 days if not optimized
        
        # Use pre-calculated volatility if available
        if hasattr(self, 'vol_63d'):
            df['Rolling_Vol'] = self.vol_63d
        else:
            # Calculate rolling volatility (fallback)
            df['Rolling_Vol'] = df['SP500_Return'].rolling(window=vol_lookback, min_periods=21).std() * np.sqrt(252)
            df['Rolling_Vol'] = df['Rolling_Vol'].fillna(df['Rolling_Vol'].mean())
        
        # Apply floor and ceiling to volatility if optimized params available
        if 'vol_floor' in params and 'vol_ceiling' in params:
            df['Rolling_Vol'] = df['Rolling_Vol'].clip(params['vol_floor'], params['vol_ceiling'])
        
        # Calculate volatility ratio for scaling positions
        df['Vol_Ratio'] = target_vol / df['Rolling_Vol']
        
        # Calculate base allocation using market predictions with optimized values
        if 'Predicted_Market' in df.columns:
            # Use optimized allocations if available
            bull_alloc = params.get('base_bull_allocation', 1.0)
            static_alloc = params.get('base_static_allocation', 0.5)
            bear_alloc = params.get('base_bear_allocation', 0.0)
            
            df['Base_SP500_Allocation'] = np.where(df['Predicted_Market'] == 'Bull', bull_alloc,
                                           np.where(df['Predicted_Market'] == 'Static', static_alloc, bear_alloc))
        elif all(col in df.columns for col in ['Bull_Prob', 'Bear_Prob']):
            # Use probability differential with floor/ceiling
            df['Base_SP500_Allocation'] = (df['Bull_Prob'] - df['Bear_Prob'] + 0.5).clip(0, 1)
        else:
            # Default to 50/50 if no predictions are available
            df['Base_SP500_Allocation'] = 0.5
            logging.warning("No prediction data found. Using 50/50 base allocation.")
            
        # Apply volatility targeting but ensure we never exceed 100% allocation
        df['Target_SP500_Allocation'] = df['Base_SP500_Allocation'] * df['Vol_Ratio']
        
        # Apply max leverage constraint - ensure no more than max_leverage
        df['SP500_Allocation'] = df['Target_SP500_Allocation'].clip(0, max_leverage)
        
        # Always ensure 100% allocation - bonds get whatever is not in stocks
        df = self._verify_allocation_totals(df)
            
        # Initialize portfolio values
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
        
        # Calculate portfolio returns
        for i in range(1, len(df)):
            # Use previous day's allocation but today's returns
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            portfolio_return = sp500_contrib + bond_contrib
            
            # Check for NaN or Inf values in portfolio return
            if not np.isfinite(portfolio_return):
                logging.warning(f"Non-finite portfolio return detected on {df.index[i]}. Setting to 0.")
                portfolio_return = 0.0  # Set to 0 to avoid cascading NaNs
            
            df.loc[df.index[i], 'Portfolio_Return'] = portfolio_return
            
            # Calculate portfolio value
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + portfolio_return)
            
            # Check for NaN or non-positive portfolio value
            if not np.isfinite(df.loc[df.index[i], 'Portfolio_Value']) or df.loc[df.index[i], 'Portfolio_Value'] <= 0:
                logging.error(f"Invalid portfolio value detected on {df.index[i]}. Resetting to a safe value.")
                df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] if i > 0 else self.initial_capital
                if df.loc[df.index[i], 'Portfolio_Value'] <= 0:
                    df.loc[df.index[i], 'Portfolio_Value'] = self.initial_capital * 0.1  # Minimal value
        
        # Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Check for NaN in Cumulative_Return and Drawdown
        if df['Cumulative_Return'].isnull().any() or df['Drawdown'].isnull().any():
            logging.warning("NaN values detected in Cumulative_Return or Drawdown. Filling with 0.")
            df['Cumulative_Return'] = df['Cumulative_Return'].fillna(0)
            df['Drawdown'] = df['Drawdown'].fillna(0)
        
        # Log allocation statistics
        full_equity_days = (df['SP500_Allocation'] >= 0.99).sum()
        full_bonds_days = (df['SP500_Allocation'] <= 0.01).sum()
        balanced_days = ((df['SP500_Allocation'] > 0.4) & (df['SP500_Allocation'] < 0.6)).sum()
        logging.info(f"Tactical allocations: {full_equity_days} days ≥99% equity, " +
                   f"{full_bonds_days} days ≤1% equity, {balanced_days} days balanced")
        
        # Store results
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return', 
            'Rolling_Vol', 'SP500_Allocation', 'Bond_Allocation',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        # Add prediction columns if available
        if 'Predicted_Market' in df.columns:
            result_columns.insert(5, 'Predicted_Market')
            
        self.results['tactical_risk_managed'] = df[result_columns]
        return self.results['tactical_risk_managed']
        
    def run_regime_adaptive_strategy(self):
        """
        Run a regime-adaptive strategy that adjusts allocations based on 
        identified market regimes. Ensures 100% allocation at all times.
        
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('regime_adaptive')
            if params:
                logging.info(f"Using optimized parameters for regime-adaptive strategy")
        
        # Standard implementation continues here
        logging.info("Running regime-adaptive strategy")
        
        # Create working copy of data
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use accurate bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Identify market regimes using multiple indicators
        # 1. Calculate moving averages
        for window in [50, 200]:
            df[f'MA_{window}'] = df['SP500'].rolling(window=window, min_periods=window//2).mean()
        
        # 2. Calculate volatility regimes
        df['Vol_21d'] = df['SP500_Return'].rolling(window=21).std() * np.sqrt(252)
        df['Vol_63d'] = df['SP500_Return'].rolling(window=63).std() * np.sqrt(252)
        df['Vol_Ratio'] = df['Vol_21d'] / df['Vol_63d'].replace(0, np.nan)
        df['Vol_Ratio'] = df['Vol_Ratio'].fillna(1.0)
        
        # 3. Define regimes using optimized parameters if available
        # Trend regime: above/below 200-day MA with minimum threshold
        uptrend_threshold = params.get('uptrend_threshold', 0.0)  # Default to 0 if not optimized
        df['Trend_Regime'] = np.where(df['SP500'] > df['MA_200'] * (1 + uptrend_threshold), 'Uptrend', 'Downtrend')
        
        # Momentum regime: 50-day MA vs 200-day MA
        df['Momentum_Regime'] = np.where(df['MA_50'] > df['MA_200'], 'Positive', 'Negative')
        
        # Volatility regime with optimized thresholds if available
        vol_expansion = params.get('vol_expansion_threshold', 1.2)
        vol_contraction = params.get('vol_contraction_threshold', 0.8)
        
        df['Vol_Regime'] = np.where(df['Vol_Ratio'] > vol_expansion, 'Expanding',
                            np.where(df['Vol_Ratio'] < vol_contraction, 'Contracting', 'Normal'))
                            
        # 4. Combined regime identification
        df['Market_Regime'] = 'Neutral'  # Default
        
        # Bullish: Uptrend + Positive Momentum + Normal/Contracting Vol
        bullish = (df['Trend_Regime'] == 'Uptrend') & (df['Momentum_Regime'] == 'Positive') & (df['Vol_Regime'] != 'Expanding')
        df.loc[bullish, 'Market_Regime'] = 'Bullish'
        
        # Bearish: Downtrend + Negative Momentum
        bearish = (df['Trend_Regime'] == 'Downtrend') & (df['Momentum_Regime'] == 'Negative')
        df.loc[bearish, 'Market_Regime'] = 'Bearish'
        
        # Volatile: Expanding Vol with either Downtrend or Negative Momentum
        volatile = (df['Vol_Regime'] == 'Expanding') & ((df['Trend_Regime'] == 'Downtrend') | (df['Momentum_Regime'] == 'Negative'))
        df.loc[volatile, 'Market_Regime'] = 'Volatile'
        
        # Recovery: Uptrend starting after Downtrend
        recovery_indices = []
        recovery_period = params.get('recovery_period', 30)  # Default to 30 days if not optimized
        
        for i in range(1, len(df)):
            if df['Trend_Regime'].iloc[i] == 'Uptrend' and df['Trend_Regime'].iloc[i-1] == 'Downtrend':
                # Mark next N days as recovery
                end_idx = min(i + recovery_period, len(df))
                recovery_indices.extend(range(i, end_idx))
        
        df.loc[df.index[recovery_indices], 'Market_Regime'] = 'Recovery'
        
        # 5. Define allocations for each regime using optimized values if available
        bullish_alloc = params.get('bullish_allocation', 1.0)
        neutral_alloc = params.get('neutral_allocation', 0.6)
        recovery_alloc = params.get('recovery_allocation', 0.8)
        volatile_alloc = params.get('volatile_allocation', 0.0)
        bearish_alloc = params.get('bearish_allocation', 0.2)
        
        df['SP500_Allocation'] = np.where(df['Market_Regime'] == 'Bullish', bullish_alloc,
                                  np.where(df['Market_Regime'] == 'Neutral', neutral_alloc,
                                  np.where(df['Market_Regime'] == 'Recovery', recovery_alloc,
                                  np.where(df['Market_Regime'] == 'Volatile', volatile_alloc, bearish_alloc))))
                                  
        # Always ensure 100% allocation - verify allocations
        df = self._verify_allocation_totals(df)
        
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
        
        # Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Log regime statistics
        regimes = df['Market_Regime'].value_counts()
        logging.info(f"Market regimes: {regimes.to_dict()}")
        
        # Store results
        self.results['regime_adaptive'] = df[[
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'Market_Regime', 'SP500_Allocation', 'Bond_Allocation',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]]
        
        return self.results['regime_adaptive']
    
    def run_volatility_targeting_strategy(self, target_vol=0.10, max_leverage=1.0):
        """
        Run a volatility targeting strategy that aims to maintain consistent portfolio risk
        while ensuring 100% allocation between stocks and bonds.
        
        Args:
            target_vol (float): Target annualized volatility (decimal)
            max_leverage (float): Maximum leverage allowed (capped at 1.0)
            
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('volatility_targeting')
            if params:
                logging.info(f"Using optimized parameters for volatility targeting strategy")
                # Use provided target_vol if specified, otherwise use optimized value
                if target_vol == 0.10:  # Default value
                    target_vol = params.get('target_vol', target_vol)
                logging.info(f"Target volatility: {target_vol:.2f}")
        
        # Force max_leverage to be at most 1.0 to ensure no leverage
        max_leverage = min(max_leverage, 1.0)
        logging.info(f"Maximum allocation capped at {max_leverage*100:.0f}% to avoid leverage")
        
        # Create working copy of data
        df = self.df.copy(deep=True)
        
        # Calculate returns and volatility
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use accurate bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Use pre-calculated volatilities if available
        if hasattr(self, 'vol_21d') and hasattr(self, 'vol_63d') and hasattr(self, 'vol_126d'):
            df['Vol_21d'] = self.vol_21d
            df['Vol_63d'] = self.vol_63d
            df['Vol_126d'] = self.vol_126d
        else:
            # Calculate rolling volatility with multiple windows (fallback)
            windows = [21, 63, 126]
            for window in windows:
                df[f'Vol_{window}d'] = df['SP500_Return'].rolling(window=window, min_periods=max(5, window//4)).std() * np.sqrt(252)
        
        # Apply floor to volatility values if optimized parameters available
        min_vol = params.get('min_vol_assumption', 0.0)
        if min_vol > 0:
            for vol_col in ['Vol_21d', 'Vol_63d', 'Vol_126d']:
                df[vol_col] = df[vol_col].fillna(min_vol)
                df[vol_col] = df[vol_col].clip(lower=min_vol)
        
        # Use a blend of volatility estimates with optimized weights if available
        if 'vol_blend_weights' in params:
            weights = params['vol_blend_weights']
            df['Blended_Vol'] = (
                weights['vol_21d'] * df['Vol_21d'].fillna(0) + 
                weights['vol_63d'] * df['Vol_63d'].fillna(0) + 
                weights['vol_126d'] * df['Vol_126d'].fillna(0)
            )
        else:
            # Default weights if not optimized
            df['Blended_Vol'] = (0.5 * df['Vol_21d'].fillna(0) + 
                                0.3 * df['Vol_63d'].fillna(0) + 
                                0.2 * df['Vol_126d'].fillna(0))
        
        # Fill any remaining NaNs with a reasonable default
        df['Blended_Vol'] = df['Blended_Vol'].fillna(0.15)  # 15% annualized is a reasonable default
        
        # Calculate target allocation based on volatility 
        df['Target_Weight'] = target_vol / df['Blended_Vol']
        
        # Incorporate prediction signals if available with optimized multipliers
        if 'Predicted_Market' in df.columns:
            bull_mult = params.get('bull_signal_multiplier', 1.0)
            static_mult = params.get('static_signal_multiplier', 0.7)
            bear_mult = params.get('bear_signal_multiplier', 0.3)
            
            df['Market_Signal'] = np.where(df['Predicted_Market'] == 'Bull', bull_mult,
                                   np.where(df['Predicted_Market'] == 'Static', static_mult, bear_mult))
            df['SP500_Allocation'] = df['Target_Weight'] * df['Market_Signal']
        else:
            df['SP500_Allocation'] = df['Target_Weight']
        
        # Apply leverage constraints - cap at max_leverage (1.0) and floor at 0
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, max_leverage)
        
        # Always ensure 100% allocation - bonds get whatever is not in stocks
        df = self._verify_allocation_totals(df)
        
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
        
        # Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Calculate realized volatility
        df['Realized_Vol_21d'] = df['Portfolio_Return'].rolling(window=21).std() * np.sqrt(252)
        
        # Log volatility statistics
        mean_allocation = df['SP500_Allocation'].mean()
        mean_realized_vol = df['Realized_Vol_21d'].mean()
        logging.info(f"Volatility targeting - avg allocation: {mean_allocation:.2f}, realized vol: {mean_realized_vol:.2%}")
        
        # Store results
        self.results['volatility_targeting'] = df[[
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return', 
            'Blended_Vol', 'SP500_Allocation', 'Bond_Allocation',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown', 'Realized_Vol_21d'
        ]]
        
        return self.results['volatility_targeting']
    
    def run_market_beating_strategy(self):
        """
        Run a strategy designed to outperform the market with controlled risk
        while ensuring 100% allocation between stocks and bonds.
        
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('market_beating')
            if params:
                logging.info(f"Using optimized parameters for market-beating strategy")
        
        # Standard implementation continues here
        logging.info("Running market-beating strategy")
        
        # Create working copy of data
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use accurate bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # 1. Technical indicators
        # Moving averages
        for window in [10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['SP500'].rolling(window=window).mean()
        
        # MACD
        macd_smoothing = params.get('macd_smoothing', 9)
        df['MACD_Line'] = df['MA_10'] - df['MA_20']
        df['MACD_Signal'] = df['MACD_Line'].rolling(window=macd_smoothing).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        
        # RSI (14-day)
        delta = df['SP500'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. Extract signals
        # Trend following signals
        df['Trend_Signal'] = 0.5  # Neutral default
        
        # Strong uptrend conditions
        strong_uptrend = (df['SP500'] > df['MA_50']) & (df['MA_50'] > df['MA_200']) & (df['MACD_Hist'] > 0)
        df.loc[strong_uptrend, 'Trend_Signal'] = 1.0
        
        # Strong downtrend conditions
        strong_downtrend = (df['SP500'] < df['MA_50']) & (df['MA_50'] < df['MA_200']) & (df['MACD_Hist'] < 0)
        df.loc[strong_downtrend, 'Trend_Signal'] = 0.0
        
        # Momentum signals
        df['Momentum_Signal'] = 0.5  # Neutral default
        
        # Strong momentum conditions
        rsi_high_threshold = params.get('rsi_high_threshold', 60)
        strong_momentum = (df['RSI'] > rsi_high_threshold) & (df['SP500'] > df['MA_20'])
        df.loc[strong_momentum, 'Momentum_Signal'] = 0.9
        
        # Weak momentum conditions
        rsi_low_threshold = params.get('rsi_low_threshold', 40)
        weak_momentum = (df['RSI'] < rsi_low_threshold) & (df['SP500'] < df['MA_20'])
        df.loc[weak_momentum, 'Momentum_Signal'] = 0.1
        
        # 3. Combine with predictions if available
        if 'Predicted_Market' in df.columns:
            df['Pred_Signal'] = np.where(df['Predicted_Market'] == 'Bull', 0.9,
                               np.where(df['Predicted_Market'] == 'Static', 0.5, 0.1))
            
            # Weighted combination of signals
            pred_weight = params.get('pred_weight', 0.4)
            trend_weight = params.get('trend_weight', 0.4)
            momentum_weight = params.get('momentum_weight', 0.2)
            
            df['Combined_Signal'] = (
                pred_weight * df['Pred_Signal'] + 
                trend_weight * df['Trend_Signal'] + 
                momentum_weight * df['Momentum_Signal']
            )
        else:
            # Without predictions, use only technical signals
            df['Combined_Signal'] = (
                0.6 * df['Trend_Signal'] + 
                0.4 * df['Momentum_Signal']
            )
        
        # 4. Generate final allocation
        # Apply more decisive sigmoid transformation to make allocations more tactical
        sigmoid_steepness = params.get('sigmoid_steepness', 8.0)
        df['Signal_Transformed'] = 1 / (1 + np.exp(-sigmoid_steepness * (df['Combined_Signal'] - 0.5)))
        
        # Final stock allocation - strictly between 0% and 100%
        df['SP500_Allocation'] = df['Signal_Transformed'].clip(0, 1.0)
        
        # Always ensure 100% allocation - bonds get whatever is not in stocks
        df = self._verify_allocation_totals(df)
        
        # 5. Calculate portfolio returns
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0
        
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Log allocation statistics 
        above_80_pct = (df['SP500_Allocation'] > 0.8).sum()
        below_20_pct = (df['SP500_Allocation'] < 0.2).sum()
        total_days = len(df)
        logging.info(f"Market-beating allocations: {above_80_pct/total_days:.1%} days >80% equity, " +
                   f"{below_20_pct/total_days:.1%} days <20% equity")
        
        # Store results
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'SP500_Allocation', 'Bond_Allocation', 'Combined_Signal',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        if 'Predicted_Market' in df.columns:
            result_columns.insert(4, 'Predicted_Market')
            
        self.results['market_beating'] = df[result_columns]
        
        return self.results['market_beating']
    
    def run_combined_anomaly_regime_strategy(self, prediction_col='Predicted_Market', anomaly_col='ensemble_anomaly'):
        """
        Run a combined strategy that integrates anomaly detection, market regimes,
        and predictions for superior performance, ensuring 100% allocation between stocks and bonds.
        
        Args:
            prediction_col (str): Column with market state predictions
            anomaly_col (str): Column with anomaly flags (-1 for anomalies, 1 for normal)
            
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('combined_anomaly_regime')
            if params:
                logging.info(f"Using optimized parameters for combined anomaly-regime strategy")
        
        # Standard implementation continues here
        logging.info("Running combined anomaly-regime strategy")
        
        # Verify required columns exist
        if prediction_col not in self.df.columns:
            logging.warning(f"Prediction column {prediction_col} not found")
            if 'Market_State' in self.df.columns:
                prediction_col = 'Market_State'
                logging.info(f"Using Market_State column instead")
            else:
                logging.warning("No prediction column found. Using 50/50 allocation.")
        
        if anomaly_col not in self.df.columns:
            logging.warning(f"Anomaly column {anomaly_col} not found. Creating dummy version.")
            # Create simple anomaly detection based on volatility spikes
            temp_df = self.df.copy()
            temp_df['SP500_Return'] = temp_df['SP500'].pct_change()
            temp_df['Return_Z'] = (temp_df['SP500_Return'] - temp_df['SP500_Return'].rolling(63).mean()) / \
                                 temp_df['SP500_Return'].rolling(63).std()
            temp_df[anomaly_col] = 1  # Default: normal
            temp_df.loc[temp_df['Return_Z'].abs() > 3, anomaly_col] = -1  # Anomaly: beyond 3 std dev
            anomaly_col = 'dummy_anomaly'
            self.df[anomaly_col] = temp_df[anomaly_col]
        
        # Create working copy
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use accurate bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # 1. Calculate market regimes
        # Moving averages
        trend_lookback = params.get('trend_lookback', {'short': 20, 'medium': 50, 'long': 200})
        for window in [trend_lookback['short'], trend_lookback['medium'], trend_lookback['long']]:
            df[f'MA_{window}'] = df['SP500'].rolling(window=window, min_periods=window//2).mean()
        
        # Volatility 
        df['Vol_21d'] = df['SP500_Return'].rolling(window=21).std() * np.sqrt(252)
        df['Vol_63d'] = df['SP500_Return'].rolling(window=63).std() * np.sqrt(252)
        df['Vol_Ratio'] = df['Vol_21d'] / df['Vol_63d'].replace(0, 0.001)
        
        # Define regimes
        short_ma = trend_lookback['short']
        medium_ma = trend_lookback['medium']
        long_ma = trend_lookback['long']
        
        df['Trend_Regime'] = 'Neutral'
        df.loc[(df['SP500'] > df[f'MA_{medium_ma}']) & (df[f'MA_{medium_ma}'] > df[f'MA_{long_ma}']), 'Trend_Regime'] = 'Bullish'
        df.loc[(df['SP500'] < df[f'MA_{medium_ma}']) & (df[f'MA_{medium_ma}'] < df[f'MA_{long_ma}']), 'Trend_Regime'] = 'Bearish'
        
        df['Vol_Regime'] = 'Normal'
        df.loc[df['Vol_Ratio'] > 1.5, 'Vol_Regime'] = 'High'
        df.loc[df['Vol_Ratio'] < 0.75, 'Vol_Regime'] = 'Low'
        
        # 2. Generate base allocation from predictions
        if prediction_col in df.columns:
            df['Base_Allocation'] = np.where(df[prediction_col] == 'Bull', 1.0,
                                     np.where(df[prediction_col] == 'Static', 0.5, 0.0))
        else:
            df['Base_Allocation'] = 0.5
        
        # 3. Apply anomaly adjustments if anomaly is detected
        df['SP500_Allocation'] = df['Base_Allocation'].copy()
        df['Anomaly_Adjustment'] = 0.0
        
        # Track anomalies and their impact
        anomalies = df[df[anomaly_col] == -1].index
        if len(anomalies) > 0:
            logging.info(f"Found {len(anomalies)} anomalies in the data")
            
            # Define recovery period after anomalies
            recovery_period = params.get('anomaly_recovery_period', 10)
            
            # Process each anomaly date
            for anomaly_date in anomalies:
                idx = df.index.get_loc(anomaly_date)
                
                # Complete risk-off on anomaly day (100% bonds)
                min_allocation = params.get('min_allocation', 0.0)
                df.loc[anomaly_date, 'SP500_Allocation'] = min_allocation
                df.loc[anomaly_date, 'Anomaly_Adjustment'] = -df['Base_Allocation'].loc[anomaly_date]
                
                # Gradual recovery over next days
                recovery_end_idx = min(idx + recovery_period, len(df)-1)
                for i in range(idx+1, recovery_end_idx+1):
                    if i < len(df):
                        # Skip if this is another anomaly day
                        if df.index[i] in anomalies:
                            continue
                            
                        # Calculate recovery factor (0 to 1)
                        days_passed = i - idx
                        recovery_factor = days_passed / recovery_period
                        
                        # Determine allocation based on recovery factor
                        original_alloc = df['Base_Allocation'].iloc[i]
                        reduced_alloc = original_alloc * recovery_factor
                        df.loc[df.index[i], 'SP500_Allocation'] = reduced_alloc
                        df.loc[df.index[i], 'Anomaly_Adjustment'] = reduced_alloc - original_alloc
        
        # 4. Apply regime conditions to fine-tune allocations
        # Reduce exposure in bearish trend regimes
        bearish_regime = (df['Trend_Regime'] == 'Bearish')
        high_vol_regime = (df['Vol_Regime'] == 'High') & (df['SP500_Allocation'] > 0.2)
        
        # Reduce stock allocation in bearish regimes by up to 50%
        bearish_reduction = params.get('bearish_reduction', 0.5)
        vol_reduction = params.get('vol_reduction', 0.8)
        
        df.loc[bearish_regime, 'SP500_Allocation'] *= bearish_reduction
        
        # Further reduce allocation during high volatility
        df.loc[high_vol_regime, 'SP500_Allocation'] *= vol_reduction
        
        # Ensure allocation is within the valid range (0-100%)
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)
        
        # Ensure 100% allocation with bonds
        df = self._verify_allocation_totals(df)
        
        # 5. Calculate portfolio returns
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        df.loc[df.index[0], 'Portfolio_Return'] = 0.0  # First day has no return
        
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # 6. Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # 7. Log strategy statistics
        anomaly_zero_alloc = ((df['SP500_Allocation'] == 0.0) & (df.index.isin(anomalies))).sum()
        avg_alloc = df['SP500_Allocation'].mean()
        logging.info(f"Combined anomaly-regime strategy: {anomaly_zero_alloc}/{len(anomalies)} anomalies with 0% equity, avg allocation: {avg_alloc:.2f}")
        
        # Store results
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'SP500_Allocation', 'Bond_Allocation', 'Trend_Regime', 'Vol_Regime',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        # Add prediction column if available
        if prediction_col in df.columns and prediction_col != 'Market_State':
            result_columns.insert(4, prediction_col)
            
        # Add anomaly column
        if anomaly_col in df.columns:
            result_columns.insert(result_columns.index('Portfolio_Return'), anomaly_col)
        
        self.results['combined_anomaly_regime'] = df[result_columns]
        return self.results['combined_anomaly_regime']
    
    def plot_regime_allocations(self, strategy_name='regime_adaptive', save_path='regime_allocations.png'):
        """
        Plot allocations by regime type
        
        Args:
            strategy_name (str): Name of strategy to analyze
            save_path (str): Path to save the plot
            
        Returns:
            None
        """
        if strategy_name not in self.results:
            logging.warning(f"Strategy '{strategy_name}' not found in results")
            return
            
        df = self.results[strategy_name]
        
        if 'Market_Regime' not in df.columns and 'Trend_Regime' not in df.columns:
            logging.warning(f"No regime information found in strategy '{strategy_name}'")
            return
            
        # Use available regime column
        regime_col = 'Market_Regime' if 'Market_Regime' in df.columns else 'Trend_Regime'
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Allocations by regime
        plt.subplot(2, 1, 1)
        
        regimes = df[regime_col].unique()
        regime_colors = {
            'Bullish': 'green',
            'Neutral': 'gray', 
            'Bearish': 'red',
            'Recovery': 'blue',
            'Volatile': 'orange'
        }
        
        # Plot equity allocations for different regimes
        for regime in regimes:
            regime_data = df[df[regime_col] == regime]
            if len(regime_data) > 0:
                color = regime_colors.get(regime, 'black')
                plt.plot(regime_data.index, regime_data['SP500_Allocation'], 
                       'o-', label=f'{regime}', color=color, alpha=0.7, markersize=3)
        
        plt.title(f'Equity Allocation by Market Regime: {strategy_name}')
        plt.ylabel('Equity Allocation %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Regime distribution and performance
        plt.subplot(2, 1, 2)
        
        # Calculate average return by regime
        regime_returns = df.groupby(regime_col)['Portfolio_Return'].mean() * 252 * 100  # Annualized %
        
        # Calculate percentage of time in each regime
        regime_counts = df[regime_col].value_counts()
        regime_pcts = regime_counts / regime_counts.sum() * 100
        
        # Create a DataFrame for plotting
        regime_stats = pd.DataFrame({
            'Annualized_Return': regime_returns,
            'Percentage_of_Time': regime_pcts
        })
        
        # Plot as a bar chart
        ax1 = plt.gca()
        regime_stats['Annualized_Return'].plot(kind='bar', position=1, width=0.3, color='blue', ax=ax1)
        ax1.set_ylabel('Annualized Return %', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add percentage of time as a second axis
        ax2 = ax1.twinx()
        regime_stats['Percentage_of_Time'].plot(kind='bar', position=0, width=0.3, color='green', ax=ax2)
        ax2.set_ylabel('% of Time', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        plt.title('Performance by Market Regime')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Regime allocations plot saved to {save_path}")
    
    def run_markov_chain_strategy(self, state_column='Market_State', training_window=252):
        """
        Run a Markov chain-based prediction strategy
        
        Args:
            state_column (str): Column with market states
            training_window (int): Days to use for training window
            
        Returns:
            DataFrame: Strategy performance
        """
        # Get optimized parameters if available and enabled
        params = {}
        if self.use_optimized and OPTIMIZATIONS_AVAILABLE:
            params = optimizer.get_parameters('markov_chain')
            if params:
                logging.info(f"Using optimized parameters for Markov chain strategy")
        
        # Standard implementation continues here
        try:
            from src.markov_strategy import MarkovStrategy
            logging.info("Running Markov Chain strategy")
        except ImportError:
            logging.error("MarkovStrategy module not available")
            return None
            
        # Create the markov strategy object
        markov = MarkovStrategy(self.df)
        
        # Train model and generate predictions
        markov.train_model(state_column=state_column, training_window=training_window)
        predictions = markov.generate_predictions()
        
        # Create working copy of data
        df = predictions.copy()
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change().fillna(0)
        
        # Use accurate bond returns if available
        if self.bond_returns is not None:
            df['Daily_Bond_Return'] = self.bond_returns
        else:
            df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Define allocations based on next state prediction
        bull_alloc = params.get('bull_allocation', 1.0)
        static_alloc = params.get('static_allocation', 0.5)
        bear_alloc = params.get('bear_allocation', 0.0)
        
        df['SP500_Allocation'] = np.where(df['Next_State_Prediction'] == 'Bull', bull_alloc,
                                 np.where(df['Next_State_Prediction'] == 'Static', static_alloc, 
                                         bear_alloc))
        
        # Scale by confidence if available
        if 'Prediction_Confidence' in df.columns:
            min_confidence = params.get('min_confidence', 0.5)
            transition_smoothing = params.get('transition_smoothing', 0.1)
            
            df['Prediction_Confidence'] = df['Prediction_Confidence'].clip(min_confidence, 1.0)
            df['SP500_Allocation'] = df['SP500_Allocation'] * (1 - transition_smoothing + transition_smoothing * df['Prediction_Confidence'])
            
        # Ensure allocation range is valid
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.0)
        
        # Ensure 100% allocation with bonds
        df = self._verify_allocation_totals(df)
        
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
        
        # Calculate performance metrics
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        df['Drawdown'] = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
        
        # Log prediction accuracy
        accuracy = (df['Next_State_Prediction'] == df[state_column].shift(-1)).mean()
        logging.info(f"Markov strategy next-state prediction accuracy: {accuracy:.2%}")
        
        # Store results
        result_columns = [
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'Next_State_Prediction', 'SP500_Allocation', 'Bond_Allocation',
            'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return', 'Drawdown'
        ]
        
        if 'Prediction_Confidence' in df.columns:
            result_columns.insert(5, 'Prediction_Confidence')
        
        self.results['markov_chain'] = df[result_columns]
        return self.results['markov_chain']
    
    def run_yield_aware_strategy(self, yield_signal_col='yield_signal', prediction_col='Predicted_Market'):
        """
        Run a strategy that combines market predictions with bond yield signals
        
        Args:
            yield_signal_col (str): Column containing yield signals between -1 and 1
            prediction_col (str): Column containing market state predictions
            
        Returns:
            DataFrame: Strategy performance metrics
        """
        logging.info("Running yield-aware investment strategy")
        
        # Get strategy parameters
        params = {}
        if self.use_optimized:
            from config.strategy_optimizations import StrategyOptimizer
            optimizer = StrategyOptimizer()
            params = optimizer.get_parameters('yield_aware')
            if params:
                logging.info("Using optimized parameters for yield-aware strategy")
        
        # Default parameters if not optimized
        bull_allocation = params.get('bull_allocation', 0.90)
        bear_allocation = params.get('bear_allocation', 0.20)
        static_allocation = params.get('static_allocation', 0.60)
        yield_impact = params.get('yield_impact', 0.3)  # How much yield signal affects allocation
        
        # Initialize strategy
        df = self.data.copy()
        df['yield_allocation'] = 0.5  # Default mid-point allocation
        
        # First, set allocations based on market state predictions
        df.loc[df[prediction_col] == 'Bull', 'yield_allocation'] = bull_allocation
        df.loc[df[prediction_col] == 'Bear', 'yield_allocation'] = bear_allocation
        df.loc[df[prediction_col] == 'Static', 'yield_allocation'] = static_allocation
        
        # Then adjust allocation based on yield signal
        if yield_signal_col in df.columns:
            # Adjust allocation up or down by yield_impact * yield_signal
            # yield_signal is between -1 and 1, so this adjusts by at most yield_impact
            df['yield_allocation'] = df['yield_allocation'] + (df[yield_signal_col] * yield_impact)
            
            # Ensure allocation stays within bounds
            df['yield_allocation'] = df['yield_allocation'].clip(0.0, 1.0)
            
            logging.info(f"Yield signals summary: mean={df[yield_signal_col].mean():.2f}, " +
                         f"min={df[yield_signal_col].min():.2f}, max={df[yield_signal_col].max():.2f}")
            
        # Run the strategy
        df['market_return'] = df['SP500'].pct_change()
        df['strategy_allocation'] = df['yield_allocation'].shift(1)
        df['strategy_return'] = df['market_return'] * df['strategy_allocation']
        df['cash_allocation'] = 1 - df['strategy_allocation']
        
        if 'BondRate' in df.columns:
            df['bond_daily_return'] = df['BondRate'] / 252 / 100  # Approximate daily bond return
            df['strategy_return'] = df['strategy_return'] + df['bond_daily_return'] * df['cash_allocation']
        
        # Calculate cumulative returns
        df['strategy_value'] = (1 + df['strategy_return']).cumprod() * self.initial_capital
        
        # Store results
        self.results['yield_aware'] = df
        
        # Calculate and log performance metrics
        total_return = (df['strategy_value'].iloc[-1] / self.initial_capital - 1) * 100
        sharpe_ratio = self.calculate_sharpe_ratio(df['strategy_return'])
        max_drawdown = self.calculate_max_drawdown(df['strategy_value'])
        win_rate = (df['strategy_return'] > 0).mean() * 100
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        
        self.log_strategy_performance('Yield Aware', total_return, annualized_return, 
                                      sharpe_ratio, max_drawdown, win_rate)
        
        return df