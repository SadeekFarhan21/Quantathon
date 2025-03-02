import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from .backtest import StrategyBacktester

class EnhancedStrategyBacktester(StrategyBacktester):
    """Enhanced backtest strategies with risk management features"""
    
    def run_volatility_targeting_strategy(self, prediction_col='Predicted_Market', target_vol=0.10, max_allocation=1.5):
        """
        Run a strategy that targets constant volatility while using prediction signals
        
        Args:
            prediction_col (str): Column with market predictions
            target_vol (float): Target annualized volatility (e.g., 0.10 = 10%)
            max_allocation (float): Maximum allocation to equities (e.g., 1.5 = 150%)
            
        Returns:
            DataFrame: Strategy performance
        """
        logging.info(f"Running volatility-targeting strategy (target: {target_vol*100:.1f}%)")
        
        # Create a copy of the data
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change()
        df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Calculate rolling volatility using a 21-day window (about 1 month)
        df['Rolling_Vol_21d'] = df['SP500_Return'].rolling(window=21).std() * np.sqrt(252)
        # And a longer window for more stability
        df['Rolling_Vol_63d'] = df['SP500_Return'].rolling(window=63).std() * np.sqrt(252)
        
        # Use a blend of short and long-term volatility estimates
        df['Blended_Vol'] = 0.7 * df['Rolling_Vol_21d'] + 0.3 * df['Rolling_Vol_63d']
        df['Blended_Vol'] = df['Blended_Vol'].fillna(df['SP500_Return'].std() * np.sqrt(252))
        
        # Base allocation on the bull/bear prediction
        df['Base_Allocation'] = np.where(df[prediction_col] == 'Bull', 1.0,
                                 np.where(df[prediction_col] == 'Static', 0.5, 0.2))
        
        # Calculate allocation to maintain target volatility
        # Formula: Allocation = Target Vol / Current Vol
        df['Vol_Allocation'] = target_vol / df['Blended_Vol']
        
        # Compute final allocation as scaled base allocation
        df['SP500_Allocation'] = df['Base_Allocation'] * df['Vol_Allocation']
        
        # Apply constraints (both minimum and maximum)
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, max_allocation)
        
        # Add a reversal signal during extreme volatility events
        volatility_spike = df['Rolling_Vol_21d'] > 2 * df['Rolling_Vol_63d']
        recent_drawdown = (df['SP500'] / df['SP500'].rolling(window=10).max() - 1) < -0.05
        
        # Reduce equity allocation during volatility spikes with drawdowns
        df.loc[volatility_spike & recent_drawdown, 'SP500_Allocation'] *= 0.5
        
        # Balance with bonds
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Calculate portfolio returns and value
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        
        # Calculate portfolio performance
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        
        # Store results
        self.results['vol_targeting'] = df[[
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'Rolling_Vol_21d', 'Rolling_Vol_63d', 'SP500_Allocation', 
            'Bond_Allocation', 'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return'
        ]]
        
        return self.results['vol_targeting']
    
    def run_tactical_momentum_strategy(self, prediction_col='Predicted_Market'):
        """
        Run a tactical momentum strategy that combines predictions with momentum signals
        
        Args:
            prediction_col (str): Column with market predictions
            
        Returns:
            DataFrame: Strategy performance
        """
        logging.info("Running tactical momentum strategy")
        
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change()
        df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Calculate momentum features
        lookback_periods = [20, 60, 120, 252]
        
        for period in lookback_periods:
            # Calculate returns over different periods
            df[f'Return_{period}d'] = df['SP500'].pct_change(periods=period)
            
            # Calculate whether price is above moving average
            df[f'Above_MA_{period}d'] = (df['SP500'] > df['SP500'].rolling(window=period).mean()).astype(float)
        
        # CREATE MULTI-TIMEFRAME MOMENTUM SIGNAL
        # This combines momentum across multiple timeframes
        df['Momentum_Score'] = (
            0.4 * df['Above_MA_20d'] +
            0.3 * df['Above_MA_60d'] +
            0.2 * df['Above_MA_120d'] +
            0.1 * df['Above_MA_252d']
        )
        
        # Add rate of change for momentum indicator
        df['RoC_20d'] = (df['SP500'] / df['SP500'].shift(20) - 1) * 100
        
        # Combine prediction with momentum score
        df['Base_Allocation'] = np.where(df[prediction_col] == 'Bull', 0.8, 
                                 np.where(df[prediction_col] == 'Static', 0.5, 0.2))
        
        # Adjust allocation based on momentum
        df['Momentum_Adj'] = df['Momentum_Score'] * 0.4  # Scale momentum impact
        
        # Apply Rate-of-Change adjustment only in specific cases
        roc_threshold = 8.0  # 8% in 20 days is strong momentum
        
        # Increase allocation for strong upward momentum in bull markets
        df.loc[(df['RoC_20d'] > roc_threshold) & (df[prediction_col] == 'Bull'), 'Momentum_Adj'] += 0.2
        
        # Decrease allocation for strong downward momentum in any market
        df.loc[df['RoC_20d'] < -roc_threshold, 'Momentum_Adj'] -= 0.3
        
        # Calculate final allocation
        df['SP500_Allocation'] = df['Base_Allocation'] + df['Momentum_Adj']
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.5)  # Allow up to 150% allocation
        
        # Bond allocation
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Calculate portfolio performance
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        
        # Store results
        self.results['tactical_momentum'] = df[[
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'Momentum_Score', 'RoC_20d', 'SP500_Allocation', 
            'Bond_Allocation', 'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return'
        ]]
        
        return self.results['tactical_momentum']
        
    def run_adaptive_strategy(self, prediction_col='Predicted_Market'):
        """
        Adaptive strategy that dynamically changes based on market regime
        
        Args:
            prediction_col (str): Column with market predictions
            
        Returns:
            DataFrame: Strategy performance
        """
        logging.info("Running adaptive market regime strategy")
        
        df = self.df.copy(deep=True)
        
        # Calculate returns
        df['SP500_Return'] = df['SP500'].pct_change()
        df['Daily_Bond_Return'] = (1 + df['BondRate'] / 100) ** (1/252) - 1
        
        # Calculate volatility and trend indicators
        df['Vol_63d'] = df['SP500_Return'].rolling(window=63).std() * np.sqrt(252)
        df['Trend_252d'] = df['SP500'].pct_change(252)
        
        # Use prediction probabilities if available for more nuanced signals
        if 'Bull_Prob' in df.columns and 'Bear_Prob' in df.columns:
            df['Signal_Strength'] = df['Bull_Prob'] - df['Bear_Prob']
            
            # Strong signal when difference between Bull and Bear probs is large
            strong_signal = abs(df['Signal_Strength']) > 0.4
        else:
            # Default to prediction certainty based on previous market state
            df['Signal_Strength'] = 0.0
            df.loc[df[prediction_col] == df[prediction_col].shift(1), 'Signal_Strength'] = 0.4
            strong_signal = True
        
        # Identify market regimes
        # 1. Low volatility bull (ideal for leverage)
        # 2. High volatility bull (reduce leverage)
        # 3. Low volatility bear (some exposure still appropriate)
        # 4. High volatility bear (minimum exposure)
        
        vol_threshold = 0.20  # 20% annualized volatility is high
        
        df['Market_Regime'] = None
        
        # Define market regimes
        df.loc[(df[prediction_col] == 'Bull') & (df['Vol_63d'] <= vol_threshold), 'Market_Regime'] = 'Low_Vol_Bull'
        df.loc[(df[prediction_col] == 'Bull') & (df['Vol_63d'] > vol_threshold), 'Market_Regime'] = 'High_Vol_Bull'
        df.loc[(df[prediction_col] == 'Bear') & (df['Vol_63d'] <= vol_threshold), 'Market_Regime'] = 'Low_Vol_Bear'
        df.loc[(df[prediction_col] == 'Bear') & (df['Vol_63d'] > vol_threshold), 'Market_Regime'] = 'High_Vol_Bear'
        df.loc[df['Market_Regime'].isnull(), 'Market_Regime'] = 'Static'
        
        # Set base allocations by regime
        base_allocations = {
            'Low_Vol_Bull': 1.3,    # Leverage in ideal conditions
            'High_Vol_Bull': 0.8,   # Reduced exposure in volatile bull markets
            'Low_Vol_Bear': 0.3,    # Some exposure in less volatile bear markets
            'High_Vol_Bear': 0.0,   # No exposure in highly volatile bear markets
            'Static': 0.5           # Balanced in static markets
        }
        
        df['Base_Allocation'] = df['Market_Regime'].map(base_allocations)
        
        # Apply signal strength adjustments
        df['Signal_Adj'] = df['Signal_Strength'].abs() * 0.3  # Scale by confidence
        
        # Apply higher allocation when signals align with trend
        trend_aligned = (df['Trend_252d'] > 0) & (df[prediction_col] == 'Bull') | \
                        (df['Trend_252d'] < 0) & (df[prediction_col] == 'Bear')
                        
        df.loc[trend_aligned, 'Signal_Adj'] += 0.2
        
        # Final stock allocation with adjustments
        df['SP500_Allocation'] = np.where(
            df[prediction_col] == 'Bull',
            df['Base_Allocation'] + df['Signal_Adj'],
            df['Base_Allocation'] - df['Signal_Adj']
        )
        
        # Apply constraints
        df['SP500_Allocation'] = df['SP500_Allocation'].clip(0, 1.5)
        df['Bond_Allocation'] = 1.0 - df['SP500_Allocation']
        
        # Calculate portfolio returns
        df['Portfolio_Return'] = np.nan
        df['Portfolio_Value'] = np.nan
        df.loc[df.index[0], 'Portfolio_Value'] = self.initial_capital
        
        for i in range(1, len(df)):
            sp500_contrib = df['SP500_Allocation'].iloc[i-1] * df['SP500_Return'].iloc[i]
            bond_contrib = df['Bond_Allocation'].iloc[i-1] * df['Daily_Bond_Return'].iloc[i]
            df.loc[df.index[i], 'Portfolio_Return'] = sp500_contrib + bond_contrib
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value'] * (1 + df.loc[df.index[i], 'Portfolio_Return'])
        
        # Calculate cumulative return
        df['Cumulative_Return'] = df['Portfolio_Value'] / self.initial_capital - 1
        
        # Store results
        self.results['adaptive_regime'] = df[[
            'SP500', 'BondRate', 'SP500_Return', 'Daily_Bond_Return',
            'Vol_63d', 'Market_Regime', 'SP500_Allocation', 
            'Bond_Allocation', 'Portfolio_Return', 'Portfolio_Value', 'Cumulative_Return'
        ]]
        
        return self.results['adaptive_regime']