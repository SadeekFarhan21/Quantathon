import numpy as np
import pandas as pd
import logging

class TacticalRiskManager:
    """
    Advanced risk management system for market-timing strategies.
    Implements various risk reduction techniques to protect capital during
    market stress while maximizing returns during favorable conditions.
    """
    
    def __init__(self, df, return_col='SP500_Return', alloc_col='SP500_Allocation'):
        """
        Initialize the risk manager
        
        Args:
            df (DataFrame): Market data with returns and allocation
            return_col (str): Column name for asset returns
            alloc_col (str): Column name for asset allocation
        """
        self.df = df.copy()
        self.return_col = return_col
        self.alloc_col = alloc_col
        
    def calculate_dynamic_volatility_target(self, target_vol=0.10, 
                                           vol_window=21, 
                                           smooth_window=10):
        """
        Calculate allocations that target a constant volatility
        
        Args:
            target_vol (float): Target annualized volatility (e.g., 0.10 = 10%)
            vol_window (int): Window for volatility calculation
            smooth_window (int): Window for smoothing volatility
            
        Returns:
            Series: Volatility-adjusted allocation
        """
        # Calculate rolling volatility
        rolling_vol = self.df[self.return_col].rolling(window=vol_window).std() * np.sqrt(252)
        
        # Smooth volatility to prevent overreacting
        smoothed_vol = rolling_vol.rolling(window=smooth_window).mean()
        
        # Fill NaNs with reasonable default
        smoothed_vol = smoothed_vol.fillna(rolling_vol.mean())
        
        # Avoid division by zero
        smoothed_vol = smoothed_vol.replace(0, rolling_vol.mean())
        
        # Calculate vol-targeting multiplier: target_vol / current_vol
        vol_multiplier = target_vol / smoothed_vol
        
        # Apply the multiplier to the base allocation
        vol_adjusted_alloc = self.df[self.alloc_col] * vol_multiplier
        
        return vol_adjusted_alloc
    
    def apply_dynamic_stop_loss(self, trailing_window=126, max_drawdown=-0.10):
        """
        Apply trailing stop-loss based on maximum asset drawdown
        
        Args:
            trailing_window (int): Window for calculating trailing max
            max_drawdown (float): Maximum allowed drawdown (negative number)
            
        Returns:
            Series: Stop-loss adjusted allocation
        """
        # Calculate price series from returns (simplified)
        price = (1 + self.df[self.return_col]).cumprod()
        
        # Calculate trailing max over specified window
        trailing_max = price.rolling(window=trailing_window, min_periods=1).max()
        
        # Calculate current drawdown from trailing max
        current_drawdown = price / trailing_max - 1
        
        # Create stop-loss multiplier: 1.0 when above threshold, 0.0 when below
        stop_loss_multiplier = (current_drawdown > max_drawdown).astype(float)
        
        # Apply stop-loss multiplier to allocation
        stop_adjusted_alloc = self.df[self.alloc_col] * stop_loss_multiplier
        
        return stop_adjusted_alloc
    
    def apply_trend_filter(self, fast_ma=50, slow_ma=200):
        """
        Apply trend filter based on moving average crossover
        
        Args:
            fast_ma (int): Fast moving average window
            slow_ma (int): Slow moving average window
            
        Returns:
            Series: Trend-filtered allocation
        """
        # Ensure we have SP500 price data
        if 'SP500' not in self.df.columns:
            return self.df[self.alloc_col]  # Return unmodified if no price data
        
        # Calculate moving averages
        fast = self.df['SP500'].rolling(window=fast_ma).mean()
        slow = self.df['SP500'].rolling(window=slow_ma).mean()
        
        # Create trend filter: 1.0 when fast > slow, 0.5 otherwise
        trend_multiplier = np.where(fast > slow, 1.0, 0.5)
        
        # Apply trend filter to allocation
        trend_adjusted_alloc = self.df[self.alloc_col] * trend_multiplier
        
        return trend_adjusted_alloc
    
    def apply_volatility_breakout_filter(self, window=21, vol_threshold=2.0):
        """
        Apply filter that reduces exposure during volatility breakouts
        
        Args:
            window (int): Window for calculating normal volatility
            vol_threshold (float): Threshold for volatility breakout (multiplier)
            
        Returns:
            Series: Volatility-adjusted allocation
        """
        # Calculate rolling volatility
        rolling_vol = self.df[self.return_col].rolling(window=window).std() * np.sqrt(252)
        
        # Calculate average volatility over a longer window
        avg_vol = rolling_vol.rolling(window=window*3).mean()
        
        # Fill NaNs
        avg_vol = avg_vol.fillna(rolling_vol.mean())
        
        # Calculate volatility ratio
        vol_ratio = rolling_vol / avg_vol
        
        # Create vol breakout filter: 0.5 when vol > threshold, 1.0 otherwise
        vol_breakout_multiplier = np.where(vol_ratio > vol_threshold, 0.5, 1.0)
        
        # Apply vol breakout filter to allocation
        vol_breakout_adjusted_alloc = self.df[self.alloc_col] * vol_breakout_multiplier
        
        return vol_breakout_adjusted_alloc
    
    def apply_all_risk_filters(self, target_vol=0.12, max_drawdown=-0.10, max_alloc=1.5):
        """
        Apply all risk management techniques in an integrated approach
        
        Args:
            target_vol (float): Target annualized volatility
            max_drawdown (float): Maximum allowed drawdown
            max_alloc (float): Maximum allocation cap
            
        Returns:
            Series: Risk-managed allocation
        """
        logging.info("Applying comprehensive risk management framework")
        
        # Start with existing allocation
        df = self.df.copy()
        original_alloc = df[self.alloc_col].copy()
        
        # Step 1: Apply trend filter (most important)
        trend_alloc = self.apply_trend_filter()
        
        # Step 2: Apply volatility targeting
        vol_alloc = self.calculate_dynamic_volatility_target(target_vol=target_vol)
        
        # Step 3: Apply stop-loss as a strict limit 
        stop_loss_filter = (self.apply_dynamic_stop_loss(max_drawdown=max_drawdown) > 0).astype(float)
        
        # Step 4: Apply volatility breakout filter as a scaling factor
        vol_breakout_scalar = self.apply_volatility_breakout_filter() / original_alloc
        vol_breakout_scalar = vol_breakout_scalar.fillna(1.0).replace(np.inf, 1.0)
        
        # Step 5: Combined model - start with volatility targeting
        risk_managed_alloc = vol_alloc
        
        # Then apply trend filter as a cap
        risk_managed_alloc = np.minimum(risk_managed_alloc, trend_alloc)
        
        # Apply stop-loss filter (binary)
        risk_managed_alloc = risk_managed_alloc * stop_loss_filter
        
        # Scale by volatility breakout filter
        risk_managed_alloc = risk_managed_alloc * vol_breakout_scalar
        
        # Apply final constraints
        risk_managed_alloc = np.clip(risk_managed_alloc, 0, max_alloc)
        
        # Fill NaNs with conservative value
        risk_managed_alloc = risk_managed_alloc.fillna(0.5)
        
        return risk_managed_alloc


class RiskMetrics:
    """Calculate advanced risk metrics for portfolio analysis"""
    
    @staticmethod
    def calculate_drawdowns(returns):
        """
        Calculate drawdown series from returns
        
        Args:
            returns (Series): Return series
            
        Returns:
            Series: Drawdowns series (negative values)
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        return drawdowns
    
    @staticmethod
    def calculate_cvar(returns, alpha=0.05):
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns (Series): Return series
            alpha (float): Significance level
            
        Returns:
            float: CVaR value
        """
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Calculate VaR
        var = np.percentile(returns, alpha * 100)
        
        # Calculate CVaR (average of returns below VaR)
        cvar = returns[returns <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_ulcer_index(returns, window=14):
        """
        Calculate Ulcer Index (measure of drawdown severity)
        
        Args:
            returns (Series): Return series
            window (int): Rolling window size
            
        Returns:
            Series: Ulcer Index
        """
        try:
            # Calculate prices from returns
            prices = (1 + pd.Series(returns).fillna(0)).cumprod()
            
            # Calculate rolling maximum
            roll_max = prices.rolling(window=window, min_periods=1).max()
            
            # Calculate percentage drawdown
            pct_drawdown = (prices - roll_max) / roll_max
            
            # Calculate squared drawdown
            squared_drawdown = pct_drawdown ** 2
            
            # Calculate Ulcer Index (square root of average squared drawdown)
            ulcer_index = np.sqrt(squared_drawdown.rolling(window=window, min_periods=1).mean())
            
            return ulcer_index
        except Exception as e:
            logging.error(f"Error calculating Ulcer Index: {str(e)}")
            # Return zeros with same index as returns if possible
            if hasattr(returns, 'index'):
                return pd.Series(0, index=returns.index)
            return None
