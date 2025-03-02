import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader_market import MarketDataLoader

def analyze_bond_rates(data_path, output_dir="results/bond_analysis"):
    """
    Analyze bond rates and their impact on investment strategies
    
    Args:
        data_path (str): Path to market data
        output_dir (str): Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    loader = MarketDataLoader(data_path)
    price_df, _ = loader.load_data()
    
    # Focus on bond rate data
    bond_df = price_df[['BondRate']].copy()
    
    # Basic statistics
    print("\n=== Bond Rate Analysis ===")
    print(f"Bond rate range: {bond_df['BondRate'].min():.2f}% to {bond_df['BondRate'].max():.2f}%")
    print(f"Mean bond rate: {bond_df['BondRate'].mean():.2f}%")
    print(f"Median bond rate: {bond_df['BondRate'].median():.2f}%")
    
    # Calculate daily bond returns (correct annualized percentage to daily decimal)
    bond_df['Daily_Bond_Return'] = bond_df['BondRate'] / 100 / 252
    
    # Calculate cumulative bond returns (compounding daily)
    bond_df['Cumulative_Factor'] = (1 + bond_df['Daily_Bond_Return']).cumprod()
    
    # Convert to cumulative percentage return
    bond_df['Cumulative_Return'] = bond_df['Cumulative_Factor'] - 1
    
    print(f"\nTotal bond return over period: {bond_df['Cumulative_Return'].iloc[-1]*100:.2f}%")
    print(f"Annualized bond return: {((1 + bond_df['Cumulative_Return'].iloc[-1]) ** (252/len(bond_df)) - 1)*100:.2f}%")
    
    # Plot bond rates over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(bond_df.index, bond_df['BondRate'], 'b-')
    plt.title('Short-Term Bond Rate (Annual %)')
    plt.ylabel('Bond Rate (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(bond_df.index, bond_df['Cumulative_Return'] * 100, 'g-')
    plt.title('Cumulative Bond Return (%)')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bond_rates.png'))
    
    # Compare bond and stock performance
    stock_bond_df = price_df.copy()
    stock_bond_df['SP500_Return'] = stock_bond_df['SP500'].pct_change()
    stock_bond_df['Daily_Bond_Return'] = stock_bond_df['BondRate'] / 100 / 252
    
    # Calculate cumulative returns
    stock_bond_df['SP500_Cumulative'] = (1 + stock_bond_df['SP500_Return'].fillna(0)).cumprod() - 1
    stock_bond_df['Bond_Cumulative'] = (1 + stock_bond_df['Daily_Bond_Return'].fillna(0)).cumprod() - 1
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(stock_bond_df.index, stock_bond_df['SP500_Cumulative'] * 100, 'b-', label='S&P 500')
    plt.plot(stock_bond_df.index, stock_bond_df['Bond_Cumulative'] * 100, 'g-', label='Bonds')
    plt.title('Cumulative Returns: S&P 500 vs. Bonds')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stock_vs_bonds.png'))
    
    # Calculate correlation
    correlation = stock_bond_df[['SP500_Return', 'Daily_Bond_Return']].corr()
    print(f"\nCorrelation between daily stock and bond returns: {correlation.iloc[0, 1]:.4f}")
    
    # Calculate rolling correlation
    rolling_corr = stock_bond_df['SP500_Return'].rolling(window=63).corr(stock_bond_df['Daily_Bond_Return'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr.index, rolling_corr, 'r-')
    plt.title('63-Day Rolling Correlation Between Stock and Bond Returns')
    plt.ylabel('Correlation')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stock_bond_correlation.png'))
    
    # Save processed data
    bond_df.to_csv(os.path.join(output_dir, 'bond_analysis.csv'))
    
    # Calculate risk-adjusted returns
    sp500_ret = stock_bond_df['SP500_Return'].mean() * 252  # Annualized
    sp500_vol = stock_bond_df['SP500_Return'].std() * np.sqrt(252)  # Annualized
    
    bond_ret = stock_bond_df['Daily_Bond_Return'].mean() * 252  # Annualized
    bond_vol = stock_bond_df['Daily_Bond_Return'].std() * np.sqrt(252)  # Annualized
    
    print("\n=== Risk-Adjusted Returns ===")
    print(f"S&P 500: Return = {sp500_ret*100:.2f}%, Volatility = {sp500_vol*100:.2f}%, Sharpe = {sp500_ret/sp500_vol:.2f}")
    print(f"Bonds: Return = {bond_ret*100:.2f}%, Volatility = {bond_vol*100:.2f}%, Sharpe = {bond_ret/bond_vol:.2f}")
    
    return bond_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze bond rates and returns')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output', type=str, default='results/bond_analysis', 
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    analyze_bond_rates(args.data, args.output)
