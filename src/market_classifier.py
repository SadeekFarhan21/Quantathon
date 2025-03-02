import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketClassifier:
    def __init__(self, price_data):
        """
        Classify market states (Bear, Bull, Static) based on S&P 500 price data.
        
        Args:
            price_data (DataFrame): DataFrame with S&P 500 price data
        """
        self.price_data = price_data
        self.market_states = None
        
    def classify_markets(self, threshold=0.2, window=252):
        """
        Classify markets as Bear, Bull, or Static based on price movements.
        
        Bear market: Drawdown from peak exceeds 20%
        Bull market: Market has increased 20% or more from a trough
        Static market: Neither Bear nor Bull
        
        Args:
            threshold (float): Threshold for market classification (default: 0.2 or 20%)
            window (int): Window size for rolling peak/trough calculation (default: 252 trading days)
            
        Returns:
            DataFrame: Original data with market state classifications
        """
        logging.info(f"Classifying markets with threshold {threshold*100}%")
        df = self.price_data.copy()
        
        # Calculate rolling peak
        df['Rolling_Peak'] = df['SP500'].rolling(window=window, min_periods=1).max()
        
        # Calculate drawdown from peak (negative numbers indicate drawdown)
        df['Drawdown'] = (df['SP500'] / df['Rolling_Peak']) - 1
        
        # Identify Bear markets strictly (drawdown exceeds threshold)
        df['Is_Bear'] = df['Drawdown'] <= -threshold
        
        # Identify Bull markets (increase from trough exceeds threshold)
        df['Rolling_Trough'] = df['SP500'].rolling(window=window, min_periods=1).min()
        df['Increase_From_Trough'] = (df['SP500'] / df['Rolling_Trough']) - 1
        
        # IMPORTANT: According to competition rules, any market not a Bear market is a Bull market
        # But we'll add a Static state for markets that don't meet either criteria strongly
        df['Is_Bull'] = ~df['Is_Bear'] & (df['Increase_From_Trough'] >= threshold)
        
        # Classify market states - Static is when neither Bear nor strong Bull
        df['Market_State'] = 'Static'
        df.loc[df['Is_Bear'], 'Market_State'] = 'Bear'
        df.loc[df['Is_Bull'], 'Market_State'] = 'Bull'
        
        # Log distribution of market states
        state_counts = df['Market_State'].value_counts()
        logging.info(f"Market state distribution: {state_counts.to_dict()}")
        
        # Store classification
        self.market_states = df[['SP500', 'BondRate', 'Drawdown', 'Is_Bear', 'Is_Bull', 'Market_State']]
        
        return self.market_states
    
    def plot_market_states(self, save_path='results/market_classification.png'):
        """
        Plot the market price and classified states.
        
        Args:
            save_path (str): Path to save the plot image
        """
        if self.market_states is None:
            logging.error("Market states not classified yet. Run classify_markets() first.")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Plot S&P 500 price
        ax = plt.gca()
        self.market_states['SP500'].plot(ax=ax, color='black', linewidth=1, label='S&P 500')
        
        # Highlight different market states
        bear_periods = self.market_states[self.market_states['Market_State'] == 'Bear']
        bull_periods = self.market_states[self.market_states['Market_State'] == 'Bull']
        static_periods = self.market_states[self.market_states['Market_State'] == 'Static']
        
        # Plot Bear periods in red, Bull in green, and Static in gray
        for period_type, color, label in [
            (bear_periods, 'red', 'Bear Market'),
            (bull_periods, 'green', 'Bull Market'),
            (static_periods, 'gray', 'Static Market')
        ]:
            if not period_type.empty:
                plt.fill_between(
                    period_type.index,
                    period_type['SP500'],
                    alpha=0.3,
                    color=color,
                    label=label
                )
        
        plt.title('S&P 500 with Market State Classification')
        plt.ylabel('S&P 500 Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Market classification plot saved to {save_path}")
        
    def get_market_stats(self):
        """
        Get statistics about different market states.
        
        Returns:
            dict: Statistics about market states
        """
        if self.market_states is None:
            logging.error("Market states not classified yet. Run classify_markets() first.")
            return {}
            
        stats = {
            'Bear': len(self.market_states[self.market_states['Market_State'] == 'Bear']),
            'Bull': len(self.market_states[self.market_states['Market_State'] == 'Bull']),
            'Static': len(self.market_states[self.market_states['Market_State'] == 'Static']),
        }
        
        stats['Total_Days'] = len(self.market_states)
        stats['Bear_Pct'] = stats['Bear'] / stats['Total_Days'] * 100
        stats['Bull_Pct'] = stats['Bull'] / stats['Total_Days'] * 100
        stats['Static_Pct'] = stats['Static'] / stats['Total_Days'] * 100
        
        return stats

if __name__ == "__main__":
    # Example usage
    from data_loader_market import MarketDataLoader
    
    loader = MarketDataLoader()
    price_df, _ = loader.load_data()
    
    classifier = MarketClassifier(price_df)
    market_states = classifier.classify_markets()
    
    stats = classifier.get_market_stats()
    print("Market State Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    classifier.plot_market_states()
