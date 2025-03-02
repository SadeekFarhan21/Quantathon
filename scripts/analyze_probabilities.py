import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader_market import MarketDataLoader
from src.market_classifier import MarketClassifier

def analyze_probabilities(data_path, output_dir="results/probability_analysis"):
    """
    Analyze market-based probabilities and their relationship to actual market states.
    
    Args:
        data_path (str): Path to market data
        output_dir (str): Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    loader = MarketDataLoader(data_path)
    price_df, prob_df = loader.load_data()
    merged_df = loader.merge_data()
    
    # Classify markets
    classifier = MarketClassifier(merged_df)
    market_states = classifier.classify_markets()
    
    # Check probability data availability and frequency
    prob_date_diffs = prob_df.index.to_series().diff().dt.days
    
    print("\n=== Probability Data Analysis ===")
    print(f"Probability data availability: {len(prob_df)} dates")
    print(f"Average days between probability updates: {prob_date_diffs.mean():.1f}")
    print(f"Median days between probability updates: {prob_date_diffs.median():.1f}")
    print(f"Min days between updates: {prob_date_diffs.min()}")
    print(f"Max days between updates: {prob_date_diffs.max()}")
    
    # Analyze probability values
    print("\n=== Probability Values ===")
    print(f"PrDec (Decrease) mean: {prob_df['PrDec'].mean():.4f}, min: {prob_df['PrDec'].min():.4f}, max: {prob_df['PrDec'].max():.4f}")
    print(f"PrInc (Increase) mean: {prob_df['PrInc'].mean():.4f}, min: {prob_df['PrInc'].min():.4f}, max: {prob_df['PrInc'].max():.4f}")
    print(f"Sum of probabilities mean: {(prob_df['PrDec'] + prob_df['PrInc']).mean():.4f}")
    
    # Combine with market states for analysis
    analysis_df = market_states.join(merged_df[['PrDec', 'PrInc']])
    
    # Calculate average probabilities by market state
    prob_by_state = analysis_df.groupby('Market_State')[['PrDec', 'PrInc']].mean()
    print("\n=== Average Probabilities by Market State ===")
    print(prob_by_state)
    
    # Plot probability distributions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(prob_df['PrDec'], kde=True, bins=20)
    plt.title('Distribution of PrDec (Decrease Probability)')
    plt.xlabel('Probability')
    
    plt.subplot(2, 2, 2)
    sns.histplot(prob_df['PrInc'], kde=True, bins=20)
    plt.title('Distribution of PrInc (Increase Probability)')
    plt.xlabel('Probability')
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='PrDec', y='PrInc', data=prob_df)
    plt.title('Relationship between PrDec and PrInc')
    plt.xlabel('PrDec (Decrease Probability)')
    plt.ylabel('PrInc (Increase Probability)')
    
    plt.subplot(2, 2, 4)
    sns.histplot(prob_df['PrDec'] + prob_df['PrInc'], kde=True, bins=20)
    plt.title('Distribution of Probability Sum (PrDec + PrInc)')
    plt.xlabel('Probability Sum')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'))
    
    # Plot probabilities over time with market states
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(analysis_df.index, analysis_df['PrDec'], 'r-', label='Decrease 20%+ Probability')
    plt.plot(analysis_df.index, analysis_df['PrInc'], 'g-', label='Increase 20%+ Probability')
    plt.title('Market-Based Probabilities Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(analysis_df.index, analysis_df['SP500'], 'b-', label='S&P 500')
    
    # Add background colors for different market states
    bear_periods = analysis_df[analysis_df['Market_State'] == 'Bear']
    bull_periods = analysis_df[analysis_df['Market_State'] == 'Bull']
    static_periods = analysis_df[analysis_df['Market_State'] == 'Static']
    
    for idx, row in bear_periods.iterrows():
        plt.axvspan(idx, idx + pd.Timedelta(days=1), color='red', alpha=0.2)
    
    for idx, row in bull_periods.iterrows():
        plt.axvspan(idx, idx + pd.Timedelta(days=1), color='green', alpha=0.2)
        
    plt.title('S&P 500 with Market Classifications')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(analysis_df.index, analysis_df['PrDec'] - analysis_df['PrInc'], 'k-', label='PrDec - PrInc')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Probability Difference (Decrease - Increase)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probabilities_over_time.png'))
    
    # Analyze predictive power
    # Create forward-looking windows to see if probabilities predict future market states
    lead_periods = [21, 63, 126, 252]  # ~1 month, 3 months, 6 months, 1 year
    
    predictive_results = {}
    
    for period in lead_periods:
        # Shift market state backward to align with earlier probabilities
        analysis_df[f'Future_Market_{period}d'] = analysis_df['Market_State'].shift(-period)
        
        # Calculate average probabilities for each future market state
        future_probs = analysis_df.groupby(f'Future_Market_{period}d')[['PrDec', 'PrInc']].mean()
        predictive_results[period] = future_probs
    
    # Print predictive power results
    print("\n=== Predictive Power Analysis ===")
    for period, result in predictive_results.items():
        print(f"\nAverage probabilities {period} days before market state:")
        print(result)
        
        # Calculate probability difference
        if 'Bear' in result.index and 'Bull' in result.index:
            bear_pdec = result.loc['Bear', 'PrDec']
            bear_pinc = result.loc['Bear', 'PrInc']
            bull_pdec = result.loc['Bull', 'PrDec']
            bull_pinc = result.loc['Bull', 'PrInc']
            
            print(f"Bear markets: PrDec - PrInc = {bear_pdec - bear_pinc:.4f}")
            print(f"Bull markets: PrDec - PrInc = {bull_pdec - bull_pinc:.4f}")
    
    # Plot correlation between probabilities and future market states
    plt.figure(figsize=(15, 8))
    
    for i, period in enumerate(lead_periods):
        plt.subplot(2, 2, i+1)
        
        # Create boxplots of probabilities by future market state
        data_to_plot = []
        labels = []
        
        for state in ['Bear', 'Static', 'Bull']:
            state_data = analysis_df[analysis_df[f'Future_Market_{period}d'] == state]
            if len(state_data) > 0:
                data_to_plot.append(state_data['PrDec'] - state_data['PrInc'])
                labels.append(state)
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.title(f'Probability Difference {period} Days Before Market State')
            plt.ylabel('PrDec - PrInc')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictive_power.png'))
    
    return {
        'avg_prob_by_state': prob_by_state,
        'predictive_power': predictive_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze market-based probabilities')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--output', type=str, default='results/probability_analysis', 
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    analyze_probabilities(args.data, args.output)