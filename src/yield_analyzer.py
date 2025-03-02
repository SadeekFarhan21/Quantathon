import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats

class YieldAnalyzer:
    """
    Analyze bond yields, yield curves, and their relationship 
    with market performance and investment decisions.
    """
    
    def __init__(self, data):
        """
        Initialize the YieldAnalyzer
        
        Args:
            data (DataFrame): DataFrame containing bond rate data
        """
        self.data = data
        self.yield_metrics = {}
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Prepare yield data for analysis"""
        # Make a copy of the data
        self.processed_data = self.data.copy()
        
        # Basic bond rate calculations
        if 'BondRate' in self.processed_data.columns:
            # Calculate rate changes
            self.processed_data['BondRate_Change'] = self.processed_data['BondRate'].diff()
            self.processed_data['BondRate_Change_Pct'] = self.processed_data['BondRate'].pct_change()
            
            # Calculate rolling statistics
            for window in [5, 10, 20, 60]:
                self.processed_data[f'BondRate_MA{window}'] = self.processed_data['BondRate'].rolling(window=window).mean()
                self.processed_data[f'BondRate_Std{window}'] = self.processed_data['BondRate'].rolling(window=window).std()
                
            # Calculate trend indicators
            self.processed_data['BondRate_Trend'] = np.where(
                self.processed_data['BondRate'] > self.processed_data['BondRate_MA20'],
                1,  # Rising yield trend
                np.where(
                    self.processed_data['BondRate'] < self.processed_data['BondRate_MA20'],
                    -1,  # Falling yield trend
                    0    # Neutral trend
                )
            )
            
            # Add equity-bond relationship metrics if SP500 data is available
            if 'SP500' in self.processed_data.columns:
                # Calculate rolling correlation between bond rates and equity
                self.processed_data['Bond_Equity_Corr60'] = self.processed_data['BondRate'].rolling(60).corr(self.processed_data['SP500'])
                
                # Calculate yield-equity ratio - lower values may indicate equity overvaluation
                self.processed_data['Yield_Equity_Ratio'] = self.processed_data['BondRate'] / self.processed_data['SP500'].pct_change(252).add(1)
                
        # Handle NAs
        self.processed_data = self.processed_data.fillna(method='bfill').fillna(method='ffill')
                
    def analyze_bond_yields(self):
        """
        Analyze bond yield patterns and relationships to market indicators
        
        Returns:
            dict: Yield analysis metrics
        """
        results = {}
        
        if 'BondRate' in self.processed_data.columns:
            # Basic yield statistics
            yield_stats = {
                'mean': float(self.processed_data['BondRate'].mean()),
                'median': float(self.processed_data['BondRate'].median()),
                'std_dev': float(self.processed_data['BondRate'].std()),
                'min': float(self.processed_data['BondRate'].min()),
                'max': float(self.processed_data['BondRate'].max()),
                'current': float(self.processed_data['BondRate'].iloc[-1]),
                'current_percentile': float(stats.percentileofscore(self.processed_data['BondRate'].dropna(), 
                                                             self.processed_data['BondRate'].iloc[-1]) / 100)
            }
            results['yield_stats'] = yield_stats
            
            # Trend analysis
            recent_window = min(60, len(self.processed_data))
            recent_trend = self.processed_data['BondRate_Trend'].iloc[-recent_window:].value_counts()
            
            trend_analysis = {
                'rising_days': int(recent_trend.get(1, 0)),
                'falling_days': int(recent_trend.get(-1, 0)),
                'neutral_days': int(recent_trend.get(0, 0)),
                'trend_strength': float(recent_trend.get(1, 0) - recent_trend.get(-1, 0)) / recent_window
            }
            results['trend_analysis'] = trend_analysis
            
            # Market regime based on yield trends
            if 'SP500' in self.processed_data.columns:
                # Calculate excess returns in different yield regimes
                self.processed_data['SP500_Return'] = self.processed_data['SP500'].pct_change()
                
                # Returns during rising/falling yield periods
                rising_returns = self.processed_data.loc[self.processed_data['BondRate_Trend'] == 1, 'SP500_Return']
                falling_returns = self.processed_data.loc[self.processed_data['BondRate_Trend'] == -1, 'SP500_Return']
                
                yield_impact = {
                    'rising_yield_avg_return': float(rising_returns.mean()),
                    'rising_yield_volatility': float(rising_returns.std()),
                    'falling_yield_avg_return': float(falling_returns.mean()),
                    'falling_yield_volatility': float(falling_returns.std()),
                    'yield_market_correlation': float(self.processed_data['BondRate'].corr(self.processed_data['SP500_Return']))
                }
                results['yield_market_impact'] = yield_impact
                
        self.yield_metrics = results
        return results
    
    def get_yield_signal(self):
        """
        Generate an investment signal based on yield analysis
        
        Returns:
            float: Value between -1 (bearish) and 1 (bullish)
        """
        if not self.yield_metrics:
            self.analyze_bond_yields()
            
        signal = 0.0
        
        # 1. Factor in current yield level relative to history
        if 'yield_stats' in self.yield_metrics:
            # Higher percentile = higher yields = possibly bearish
            current_percentile = self.yield_metrics['yield_stats']['current_percentile']
            # Transform from [0,1] to [-0.5,0.5] (moderate signal contribution)
            signal -= (current_percentile - 0.5)
            
        # 2. Factor in recent yield trend
        if 'trend_analysis' in self.yield_metrics:
            trend_strength = self.yield_metrics['trend_analysis']['trend_strength']
            # Transform from [-1,1] to [-0.25,0.25] (smaller contribution)
            signal -= trend_strength * 0.25
            
        # 3. Factor in yield-equity correlation
        if 'yield_market_impact' in self.yield_metrics:
            if self.yield_metrics['yield_market_impact']['yield_market_correlation'] > 0:
                # Positive correlation: rising yields = rising equity (good)
                # In this regime, we can be more bullish
                signal += 0.25
            else:
                # Negative correlation: rising yields = falling equity (bad)
                # In this regime, we should be more cautious
                signal -= 0.25
                
        # Ensure signal is within [-1, 1]
        return max(-1.0, min(1.0, signal))
        
    def generate_report(self, output_dir="results/yield_analysis"):
        """
        Generate visualizations and analysis of yield patterns
        
        Args:
            output_dir (str): Directory to save results
            
        Returns:
            dict: Analysis results
        """
        if not self.yield_metrics:
            self.analyze_bond_yields()
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        self._plot_yield_trends(output_dir)
        self._plot_yield_market_relationship(output_dir)
        
        # Save yield metrics as CSV
        results_df = pd.DataFrame()
        
        # Flatten metrics into DataFrame rows
        for category, metrics in self.yield_metrics.items():
            for metric, value in metrics.items():
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Category': [category],
                    'Metric': [metric],
                    'Value': [value]
                })])
                
        results_df.to_csv(os.path.join(output_dir, "yield_metrics.csv"), index=False)
        
        return self.yield_metrics
    
    def _plot_yield_trends(self, output_dir):
        """Generate yield trend visualizations"""
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Bond Rate and Moving Averages
        plt.subplot(2, 1, 1)
        plt.plot(self.processed_data.index, self.processed_data['BondRate'], 'b-', label='Bond Rate')
        plt.plot(self.processed_data.index, self.processed_data['BondRate_MA20'], 'r--', label='20-day MA')
        plt.plot(self.processed_data.index, self.processed_data['BondRate_MA60'], 'g--', label='60-day MA')
        
        plt.title('Bond Rate Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Bond Rate Changes
        plt.subplot(2, 1, 2)
        plt.bar(self.processed_data.index, self.processed_data['BondRate_Change'], color='blue', alpha=0.6)
        
        plt.title('Bond Rate Daily Changes')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "yield_trends.png"))
        plt.close()
        
    def _plot_yield_market_relationship(self, output_dir):
        """Visualize relationship between yields and market performance"""
        if 'SP500' not in self.processed_data.columns:
            return
            
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Bond Rate vs Equity Performance
        plt.subplot(2, 1, 1)
        
        # Normalize both series to compare trends
        bond_norm = (self.processed_data['BondRate'] - self.processed_data['BondRate'].min()) / \
                   (self.processed_data['BondRate'].max() - self.processed_data['BondRate'].min())
                   
        equity_norm = (self.processed_data['SP500'] - self.processed_data['SP500'].min()) / \
                     (self.processed_data['SP500'].max() - self.processed_data['SP500'].min())
        
        plt.plot(self.processed_data.index, bond_norm, 'b-', label='Bond Rate (Normalized)')
        plt.plot(self.processed_data.index, equity_norm, 'r-', label='S&P 500 (Normalized)')
        
        plt.title('Bond Rate vs Equity Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Correlation
        plt.subplot(2, 1, 2)
        plt.plot(self.processed_data.index, self.processed_data['Bond_Equity_Corr60'], 'g-')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        plt.title('60-day Rolling Correlation: Bond Rate vs S&P 500')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "yield_market_relationship.png"))
        plt.close()

    def get_allocation_advice(self):
        """
        Get allocation advice based on yield analysis
        
        Returns:
            dict: Allocation recommendations and reasoning
        """
        if not self.yield_metrics:
            self.analyze_bond_yields()
            
        signal = self.get_yield_signal()
        
        # Convert signal to allocation advice
        equity_allocation = 0.5 + (signal * 0.4)  # Transform [-1,1] to [0.1,0.9]
        bond_allocation = 1.0 - equity_allocation
        
        # Generate reasoning based on metrics
        reasons = []
        
        if 'yield_stats' in self.yield_metrics:
            current_level = self.yield_metrics['yield_stats']['current']
            current_percentile = self.yield_metrics['yield_stats']['current_percentile']
            
            if current_percentile > 0.8:
                reasons.append(f"Bond rates are high (top {current_percentile:.0%} historically)")
            elif current_percentile < 0.2:
                reasons.append(f"Bond rates are low (bottom {current_percentile:.0%} historically)")
                
        if 'trend_analysis' in self.yield_metrics:
            trend_strength = self.yield_metrics['trend_analysis']['trend_strength']
            
            if trend_strength > 0.3:
                reasons.append("Bond rates are in a strong rising trend")
            elif trend_strength < -0.3:
                reasons.append("Bond rates are in a strong falling trend")
                
        if 'yield_market_impact' in self.yield_metrics:
            corr = self.yield_metrics['yield_market_impact']['yield_market_correlation']
            
            if corr > 0.3:
                reasons.append("Bond rates and equity have strong positive correlation")
            elif corr < -0.3:
                reasons.append("Bond rates and equity have strong negative correlation")
                
        advice = {
            'signal': signal,
            'equity_allocation': equity_allocation,
            'bond_allocation': bond_allocation,
            'reasons': reasons
        }
        
        return advice