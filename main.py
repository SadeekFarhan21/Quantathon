import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the current directory to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader_market import MarketDataLoader
from src.market_classifier import MarketClassifier
from src.prediction_model import MarketPredictor
from src.backtest import StrategyBacktester
from src.yield_analyzer import YieldAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/market_prediction.log", mode='w'),
        logging.StreamHandler()
    ]
)

class AnomalyDetector:
    """Detect and analyze market anomalies and tail events"""
    
    def __init__(self, data, contamination=0.05):
        """
        Initialize anomaly detector
        
        Args:
            data (DataFrame): Market data
            contamination (float): Expected proportion of anomalies
        """
        self.data = data
        self.contamination = contamination
        self.anomalies = None
        self.models = {}
        
    def detect_anomalies(self, methods=None):
        """
        Detect anomalies using multiple methods
        
        Args:
            methods (list): List of methods to use. Available: 'isolation_forest',
                          'local_outlier_factor', 'volatility', 'jumps'
                          
        Returns:
            DataFrame: Data with anomaly scores and flags
        """
        if methods is None:
            methods = ['isolation_forest', 'volatility', 'jumps']
            
        df = self.data.copy()
        
        # Prepare features for anomaly detection
        features = ['SP500', 'BondRate']
        if 'PrDec' in df.columns and 'PrInc' in df.columns:
            features.extend(['PrDec', 'PrInc'])
            
        # Add returns and volatility features
        df['SP500_Return'] = df['SP500'].pct_change()
        df['Rolling_Vol_20d'] = df['SP500_Return'].rolling(window=20).std()
        
        # Minimum data required
        min_data_points = 100
        if len(df) < min_data_points:
            logging.warning(f"Insufficient data for anomaly detection (need {min_data_points}, have {len(df)})")
            return df
        
        # Method 1: Isolation Forest
        if 'isolation_forest' in methods:
            logging.info("Running Isolation Forest anomaly detection")
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            
            # Prepare feature matrix
            X = df[features].dropna()
            
            # Fit and predict
            model.fit(X)
            self.models['isolation_forest'] = model
            
            # Get anomaly scores (-1 for anomalies, 1 for normal)
            df.loc[X.index, 'IF_Anomaly'] = model.predict(X)
            df['IF_Score'] = np.nan
            df.loc[X.index, 'IF_Score'] = model.decision_function(X)
            
        # Method 2: Local Outlier Factor
        if 'local_outlier_factor' in methods:
            logging.info("Running Local Outlier Factor anomaly detection")
            model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination
            )
            
            # Prepare feature matrix
            X = df[features].dropna()
            
            # Fit and predict
            lof_pred = model.fit_predict(X)
            
            # Get anomaly scores (-1 for anomalies, 1 for normal)
            df.loc[X.index, 'LOF_Anomaly'] = lof_pred
            df['LOF_Score'] = np.nan
            negative_outlier_factor = model.negative_outlier_factor_
            df.loc[X.index, 'LOF_Score'] = negative_outlier_factor
            
        # Method 3: Volatility-based anomalies
        if 'volatility' in methods:
            logging.info("Detecting volatility-based anomalies")
            # Calculate rolling Z-score of volatility
            vol_mean = df['Rolling_Vol_20d'].rolling(window=60).mean()
            vol_std = df['Rolling_Vol_20d'].rolling(window=60).std()
            df['Vol_Z_Score'] = (df['Rolling_Vol_20d'] - vol_mean) / vol_std.replace(0, 1)
            
            # Identify volatility anomalies (>3 std dev)
            df['Vol_Anomaly'] = 1
            df.loc[df['Vol_Z_Score'] > 3, 'Vol_Anomaly'] = -1
            
        # Method 4: Price jump anomalies
        if 'jumps' in methods:
            logging.info("Detecting price jump anomalies")
            # Calculate returns Z-score
            ret_mean = df['SP500_Return'].rolling(window=60).mean()
            ret_std = df['SP500_Return'].rolling(window=60).std()
            df['Return_Z_Score'] = (df['SP500_Return'] - ret_mean) / ret_std.replace(0, 1)
            
            # Identify jump anomalies (>4 std dev)
            df['Jump_Anomaly'] = 1
            df.loc[abs(df['Return_Z_Score']) > 4, 'Jump_Anomaly'] = -1
            
        # Combine anomaly signals
        anomaly_cols = [col for col in df.columns if col.endswith('_Anomaly')]
        if anomaly_cols:
            df['Combined_Anomaly'] = 1
            for col in anomaly_cols:
                # If any method detects an anomaly, mark it
                df.loc[df[col] == -1, 'Combined_Anomaly'] = -1
                
        self.anomalies = df
        return df
    
    def analyze_anomalies(self, output_dir="results/anomalies"):
        """
        Analyze detected anomalies
        
        Args:
            output_dir (str): Directory to save analysis
            
        Returns:
            dict: Analysis results
        """
        if self.anomalies is None:
            self.detect_anomalies()
            
        os.makedirs(output_dir, exist_ok=True)
        df = self.anomalies
        
        # Get anomaly dates
        anomaly_dates = df[df['Combined_Anomaly'] == -1].index
        logging.info(f"Found {len(anomaly_dates)} anomalies in {len(df)} data points")
        
        # Calculate market stats around anomalies
        pre_anomaly_window = 5
        post_anomaly_window = 20
        
        anomaly_impacts = []
        
        for date in anomaly_dates:
            try:
                # Get data around the anomaly
                idx = df.index.get_loc(date)
                pre_start = max(0, idx - pre_anomaly_window)
                post_end = min(len(df) - 1, idx + post_anomaly_window)
                
                pre_data = df.iloc[pre_start:idx]
                post_data = df.iloc[idx:post_end+1]
                
                # Calculate impact metrics
                pre_return = pre_data['SP500_Return'].mean()
                post_return = post_data['SP500_Return'].mean()
                anomaly_return = df.iloc[idx]['SP500_Return']
                recovery_days = None
                
                # Find recovery (when price returns to pre-anomaly level)
                pre_price = df.iloc[pre_start]['SP500']
                anomaly_price = df.iloc[idx]['SP500']
                
                if anomaly_price < pre_price:  # Price dropped
                    recovery_idx = None
                    for i in range(idx + 1, post_end + 1):
                        if df.iloc[i]['SP500'] >= pre_price:
                            recovery_idx = i
                            break
                            
                    if recovery_idx is not None:
                        recovery_days = (df.index[recovery_idx] - date).days
                
                anomaly_impacts.append({
                    'Date': date,
                    'Return': anomaly_return,
                    'Pre_Return': pre_return,
                    'Post_Return': post_return,
                    'Recovery_Days': recovery_days,
                    'IF_Score': df.loc[date, 'IF_Score'] if 'IF_Score' in df.columns else None,
                    'Vol_Z_Score': df.loc[date, 'Vol_Z_Score'] if 'Vol_Z_Score' in df.columns else None
                })
            except Exception as e:
                logging.error(f"Error analyzing anomaly at {date}: {str(e)}")
        
        # Create anomaly impact DataFrame
        impact_df = pd.DataFrame(anomaly_impacts)
        impact_df.to_csv(os.path.join(output_dir, "anomaly_impacts.csv"))
        
        # Plot anomalies
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Price with anomalies
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['SP500'], 'b-', label='S&P 500')
        
        if len(anomaly_dates) > 0:
            plt.scatter(anomaly_dates, df.loc[anomaly_dates, 'SP500'], 
                       color='red', marker='o', s=50, label='Anomalies')
            
        plt.title('S&P 500 with Detected Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        plt.subplot(3, 1, 2)
        if 'IF_Score' in df.columns:
            plt.plot(df.index, df['IF_Score'], 'g-', label='Isolation Forest Score')
        if 'LOF_Score' in df.columns:
            plt.plot(df.index, df['LOF_Score'], 'm-', label='LOF Score')
            
        plt.title('Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Volatility Z-score
        plt.subplot(3, 1, 3)
        if 'Vol_Z_Score' in df.columns:
            plt.plot(df.index, df['Vol_Z_Score'], 'r-', label='Volatility Z-Score')
            plt.axhline(y=3, color='k', linestyle='--', alpha=0.7)
            
        plt.title('Volatility Z-Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anomalies_analysis.png"))
        plt.close()
        
        return {
            'anomaly_dates': anomaly_dates,
            'impact_analysis': impact_df
        }
        
class CatastropheModeler:
    """Model extreme market events and tail risks"""
    
    def __init__(self, returns):
        """
        Initialize catastrophe modeler
        
        Args:
            returns (Series): Series of returns
        """
        self.returns = returns
        
    def fit_extreme_value_distribution(self):
        """
        Fit extreme value distribution to the returns
        
        Returns:
            tuple: Distribution parameters and stats
        """
        try:
            from scipy import stats
            
            # Calculate negative returns for losses modeling
            losses = -self.returns[self.returns < 0]
            
            if len(losses) < 50:
                logging.warning("Insufficient loss data for extreme value modeling")
                return None
                
            # Fit Generalized Pareto Distribution
            shape, loc, scale = stats.genpareto.fit(losses)
            
            return {
                'distribution': 'genpareto',
                'shape': shape,
                'location': loc,
                'scale': scale
            }
            
        except Exception as e:
            logging.error(f"Error fitting extreme value distribution: {str(e)}")
            return None
            
    def estimate_var_es(self, confidence_levels=[0.95, 0.99, 0.999]):
        """
        Estimate Value at Risk and Expected Shortfall
        
        Args:
            confidence_levels (list): Confidence levels for VaR
            
        Returns:
            dict: VaR and ES estimates
        """
        try:
            results = {}
            
            # Historical method
            for cl in confidence_levels:
                var = np.percentile(self.returns, 100 * (1 - cl))
                es = self.returns[self.returns <= var].mean()
                
                results[f'VaR_{cl:.3f}_historical'] = var
                results[f'ES_{cl:.3f}_historical'] = es
            
            # Parametric method (if available)
            evd_params = self.fit_extreme_value_distribution()
            if evd_params:
                from scipy import stats
                
                for cl in confidence_levels:
                    # Calculate parametric VaR using fitted distribution
                    var_p = -stats.genpareto.ppf(
                        1-cl, 
                        evd_params['shape'], 
                        loc=evd_params['location'], 
                        scale=evd_params['scale']
                    )
                    
                    # Compute Expected Shortfall using numerical integration
                    alpha = 1-cl
                    k = evd_params['shape']
                    sigma = evd_params['scale']
                    
                    # For GPD with shape parameter k < 1
                    if k < 1:
                        es_p = var_p * (1 + k) / (1 - k)
                    else:
                        es_p = var_p * 2  # Rough approximation for k >= 1
                        
                    results[f'VaR_{cl:.3f}_parametric'] = var_p
                    results[f'ES_{cl:.3f}_parametric'] = es_p
            
            return results
            
        except Exception as e:
            logging.error(f"Error calculating risk measures: {str(e)}")
            return {}
            
    def stress_test(self, scenario_shocks=None):
        """
        Perform stress tests based on historical or hypothetical scenarios
        
        Args:
            scenario_shocks (dict): Dictionary of scenario shocks as return multipliers
            
        Returns:
            dict: Stress test results
        """
        if scenario_shocks is None:
            # Default stress scenarios based on historical events
            scenario_shocks = {
                'COVID_Crash': -0.30,      # 30% drop (March 2020)
                'GFC_2008': -0.45,         # 45% drop (2008-2009)
            }
            
        results = {}
        
        try:
            # Get portfolio baseline stats
            baseline_mean = self.returns.mean()
            baseline_std = self.returns.std()
            baseline_sharpe = baseline_mean / baseline_std if baseline_std > 0 else 0
            
            # Simulate each scenario
            for scenario, shock in scenario_shocks.items():
                # Create a copy of returns
                scenario_returns = self.returns.copy()
                
                # Add shock as a single extreme event
                shock_idx = len(scenario_returns) // 2  # Place shock in the middle
                scenario_returns.iloc[shock_idx] = shock
                
                # Calculate impact metrics
                post_shock_returns = scenario_returns.iloc[shock_idx:]
                recovery_days = None
                
                # Find time to recover (cumulative return > 0)
                cum_returns = (1 + post_shock_returns).cumprod() - 1
                recovery_idx = (cum_returns > 0).idxmax() if any(cum_returns > 0) else None
                
                if recovery_idx:
                    recovery_days = (recovery_idx - scenario_returns.index[shock_idx]).days
                
                # Drawdown after shock
                max_dd = (cum_returns.cummax() - cum_returns).max()
                
                results[scenario] = {
                    'Shock_Magnitude': shock,
                    'Recovery_Days': recovery_days,
                    'Max_Drawdown': max_dd,
                    'Post_Shock_Volatility': post_shock_returns.std() * np.sqrt(252)
                }
                
            return results
            
        except Exception as e:
            logging.error(f"Error in stress testing: {str(e)}")
            return {}
    
    def generate_report(self, output_dir="results/catastrophe"):
        """
        Generate a complete tail risk report
        
        Args:
            output_dir (str): Directory to save report
            
        Returns:
            dict: Complete risk report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate risk measures
        var_es_results = self.estimate_var_es()
        
        # Run stress tests
        stress_results = self.stress_test()
        
        # Get extreme value distribution parameters
        evd_params = self.fit_extreme_value_distribution()
        
        # Create consolidated report
        report = {
            'risk_measures': var_es_results,
            'stress_tests': stress_results,
            'extreme_value_model': evd_params
        }
        
        # Save report as JSON
        import json
        with open(os.path.join(output_dir, "tail_risk_report.json"), "w") as f:
            json.dump(report, f, indent=4)
        
        # Create visualizations
        self.plot_tail_risk_analysis(output_dir)
        
        return report
    
    def plot_tail_risk_analysis(self, output_dir):
        """Create visualizations for tail risk analysis"""
        try:
            from scipy import stats
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot 1: Return distribution with normal fit
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            sns.histplot(self.returns, kde=True, stat='density', bins=50)
            
            # Add normal distribution fit
            x = np.linspace(min(self.returns), max(self.returns), 100)
            mean, std = self.returns.mean(), self.returns.std()
            plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', label='Normal distribution')
            plt.title('Return Distribution with Normal Fit')
            plt.xlabel('Return')
            plt.legend()
            
            # Plot 2: QQ plot against normal distribution
            plt.subplot(2, 2, 2)
            stats.probplot(self.returns, dist="norm", plot=plt)
            plt.title('QQ Plot vs. Normal Distribution')
            
            # Plot 3: Left tail zoomed in
            plt.subplot(2, 2, 3)
            negative_returns = self.returns[self.returns < 0]
            sns.histplot(negative_returns, kde=True, stat='density', bins=30)
            plt.title('Left Tail Distribution')
            
            # Add fitted GPD if available
            evd_params = self.fit_extreme_value_distribution()
            if evd_params:
                x_tail = np.linspace(min(negative_returns), 0, 100)
                pdf_values = stats.genpareto.pdf(
                    -x_tail, 
                    evd_params['shape'],
                    loc=evd_params['location'], 
                    scale=evd_params['scale']
                )
                plt.plot(-x_tail, pdf_values, 'g-', label='Generalized Pareto Fit')
                plt.legend()
                
            # Plot 4: VaR and ES visualization
            plt.subplot(2, 2, 4)
            
            var_95 = np.percentile(self.returns, 5)
            var_99 = np.percentile(self.returns, 1)
            
            sns.histplot(self.returns, kde=False, stat='density', bins=50)
            plt.axvline(x=var_95, color='orange', linestyle='--', alpha=0.8, label='VaR 95%')
            plt.axvline(x=var_99, color='red', linestyle='--', alpha=0.8, label='VaR 99%')
            
            # Shade the tail for ES
            left_of_var = self.returns[self.returns <= var_95]
            plt.fill_between(
                np.sort(left_of_var), 
                0, 
                 0.1,  # Height of shading
                alpha=0.3, 
                color='grey',
                label='ES Region'
            )
            
            plt.title('Value at Risk and Expected Shortfall')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "tail_risk_analysis.png"))
            plt.close()
            
            # Create stress test visualization
            stress_results = self.stress_test()
            
            if stress_results:
                plt.figure(figsize=(12, 6))
                
                scenarios = list(stress_results.keys())
                shock_magnitudes = [r['Shock_Magnitude'] * 100 for r in stress_results.values()]
                recovery_days = [r['Recovery_Days'] if r['Recovery_Days'] else 0 for r in stress_results.values()]
                
                x = range(len(scenarios))
                
                plt.bar(x, shock_magnitudes)
                plt.xticks(x, scenarios, rotation=45)
                plt.title('Stress Test Scenarios - Shock Magnitude (%)')
                plt.ylabel('Shock Magnitude (%)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "stress_test_shocks.png"))
                plt.close()
                
        except Exception as e:
            logging.error(f"Error plotting tail risk analysis: {str(e)}")

def main(args):
    """Run the full market prediction and investment strategy pipeline"""
    # Ensure output directories exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check advanced model usage
    if args.advanced:
        logging.info(f"Advanced PyTorch model ({args.model_type}) enabled")
        try:
            import torch
            logging.info(f"PyTorch version: {torch.__version__}")
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logging.error("PyTorch is not installed! Advanced model option will be disabled.")
            args.advanced = False
    else:
        logging.info("Using traditional ML model (not using advanced PyTorch model)")

    # Step 1: Load and prepare market data - first load ALL available data
    logging.info(f"Loading market data from {args.data}")
    loader = MarketDataLoader(args.data)
    price_df, prob_df = loader.load_data()
    
    # Merge datasets (correctly handling probabilities)
    all_data = loader.merge_data()
    
    # Convert date parameters to datetime explicitly to ensure proper filtering
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    
    # Now create a filtered version for analysis/prediction phase if requested
    if start_date and end_date:
        logging.info(f"Analysis will be performed on data from {start_date.date()} to {end_date.date()}")
        # Make sure to properly clip the data range
        analysis_data = all_data[all_data.index >= start_date]
        analysis_data = analysis_data[analysis_data.index <= end_date]
        
        # Also filter the all_data to ensure consistency later in the code
        all_data = all_data[all_data.index <= end_date]
        
        # Double-check the date range
        if not analysis_data.empty:
            actual_start = analysis_data.index.min().date()
            actual_end = analysis_data.index.max().date()
            logging.info(f"Actual analysis period: {actual_start} to {actual_end}")
            if actual_end > end_date.date():
                logging.warning(f"WARNING: Data extends beyond specified end date: {actual_end} > {end_date.date()}")
    else:
        analysis_data = all_data
        
    logging.info(f"Total available data: {len(all_data)} days from {all_data.index.min().date()} to {all_data.index.max().date()}")
    logging.info(f"Analysis data: {len(analysis_data)} days from {analysis_data.index.min().date()} to {analysis_data.index.max().date()}")
    
    # Step 2: Classify markets for ALL data first
    logging.info("Classifying market states")
    classifier = MarketClassifier(all_data)
    all_market_states = classifier.classify_markets()
    
    # Also classify specifically for analysis period
    analysis_classifier = MarketClassifier(analysis_data)
    analysis_market_states = analysis_classifier.classify_markets()
    
    # Plot and save market classifications for analysis period
    analysis_classifier.plot_market_states(save_path=os.path.join(args.output, "market_states.png"))
    analysis_market_states.to_csv(os.path.join(args.output, "market_states.csv"))
    
    # Get market state statistics
    state_stats = analysis_classifier.get_market_stats()
    logging.info(f"Market states: {state_stats['Bear']} bear days ({state_stats['Bear_Pct']:.1f}%), " +
                f"{state_stats['Bull']} bull days ({state_stats['Bull_Pct']:.1f}%), " +
                f"{state_stats['Static']} static days ({state_stats['Static_Pct']:.1f}%)")
    
    # Step 3: Analyze probabilities
    logging.info("Analyzing market-based probabilities")
    from scripts.analyze_probabilities import analyze_probabilities
    prob_analysis = analyze_probabilities(args.data, os.path.join(args.output, "probability_analysis"))
    
    # Step 4: Analyze bond rates
    logging.info("Analyzing bond rates")
    from scripts.bond_rate_analysis import analyze_bond_rates
    bond_analysis = analyze_bond_rates(args.data, os.path.join(args.output, "bond_analysis"))
    
    # Step 5: Advanced anomaly detection
    logging.info("Running advanced anomaly detection")
    from src.market_anomaly import MarketAnomalyDetector
    
    anomaly_detector = MarketAnomalyDetector(analysis_data, contamination=0.03)  # 3% anomaly rate
    anomalies = anomaly_detector.detect_anomalies(
        methods=['isolation_forest', 'dbscan', 'statistical'],
        feature_sets=None  # Use default feature sets
    )
    
    # Make sure ensemble_anomaly is always available - this is critical!
    if 'ensemble_anomaly' not in anomalies.columns:
        logging.warning("ensemble_anomaly column not created by detector, creating dummy version")
        # Create a dummy anomaly column using volatility spikes as a simple heuristic
        sp500_returns = analysis_data['SP500'].pct_change()
        rolling_std = sp500_returns.rolling(21).std()
        anomalies['ensemble_anomaly'] = 1  # Normal by default
        # Mark extreme volatility days (>3 std dev) as anomalies (-1)
        anomalies.loc[sp500_returns.abs() > rolling_std * 3, 'ensemble_anomaly'] = -1
    
    # Store anomaly detection results for later use
    anomaly_analysis = anomaly_detector.analyze_anomalies(output_dir=os.path.join(args.output, "anomalies"))
    
    # Step 6: Perform catastrophe modeling and tail risk analysis
    logging.info("Performing catastrophe modeling and tail risk analysis")
    # Create return series for modeling
    returns = analysis_data['SP500'].pct_change().dropna()
    
    # Initialize and run catastrophe modeler
    catastrophe_modeler = CatastropheModeler(returns)
    risk_report = catastrophe_modeler.generate_report(output_dir=os.path.join(args.output, "catastrophe"))
    
    # Log tail risk metrics
    var_95 = risk_report['risk_measures'].get('VaR_0.950_historical', 0)
    es_95 = risk_report['risk_measures'].get('ES_0.950_historical', 0)
    logging.info(f"Daily VaR (95%): {var_95:.4f}, Expected Shortfall: {es_95:.4f}")
    
    # Log stress test results for one scenario
    if 'GFC_2008' in risk_report['stress_tests']:
        gfc = risk_report['stress_tests']['GFC_2008']
        logging.info(f"GFC stress test: {gfc['Shock_Magnitude']*100:.1f}% shock, {gfc['Recovery_Days'] or 'N/A'} days to recovery")
    
    # Step 7: Prepare data for training and prediction - this is where we make a clear separation
    logging.info("Preparing data for training and prediction")
    
    # For training: Use data between train_start_date and train_end_date
    # Combine market states with probability data for ALL data
    combined_data = all_market_states.join(all_data[['PrDec', 'PrInc']])
    
    # Define training data based on date ranges
    available_min = combined_data.index.min()
    available_max = combined_data.index.max()

    # Define training period 
    if args.train_end_date and args.train_start_date:
        train_start = max(pd.to_datetime(args.train_start_date), available_min)
        train_end_input = pd.to_datetime(args.train_end_date)
        if train_end_input < available_min:
            logging.warning(f"Train end date {args.train_end_date} is before available data range; adjusting to {available_min.date()}.")
            train_end = available_min
        else:
            train_end = min(train_end_input, available_max)  # Don't go beyond available data
        train_data = combined_data.loc[train_start:train_end]
        logging.info(f"Using data from {train_start.date()} to {train_end.date()} for training model")
    else:
        logging.warning("No training period specified. Using default training approach.")
        # Default is to use non-analysis data for training
        if args.start_date:
            analysis_start = pd.to_datetime(args.start_date)
            train_data = combined_data.loc[:analysis_start]
        else:
            # No analysis period either, use 80% of data for training
            split_idx = int(len(combined_data) * 0.8)
            train_data = combined_data.iloc[:split_idx]
        
    # For prediction/analysis: Use data from the analysis period
    test_data = analysis_market_states.join(analysis_data[['PrDec', 'PrInc']])
    
    # Double-check the test data date range again
    if not test_data.empty:
        test_start = test_data.index.min().date()
        test_end = test_data.index.max().date()
        logging.info(f"Test data period: {test_start} to {test_end}")
        if end_date and test_end > end_date.date():
            logging.warning(f"CRITICAL: Test data extends beyond specified end date! Fixing...")
            test_data = test_data[test_data.index <= end_date]
            logging.info(f"Corrected test data range: {test_data.index.min().date()} to {test_data.index.max().date()}")
    
    logging.info(f"Using {len(train_data)} days for training, {len(test_data)} days for analysis/prediction")
    
    # Step 8: Train prediction model and generate predictions
    logging.info("Training market prediction model")
    
    predictor = None
    
    if args.advanced:
        try:
            logging.info("Advanced mode enabled: using PyTorch-based model")
            from src.advanced_models import AdvancedMarketPredictor
            predictor = AdvancedMarketPredictor(model_type=args.model_type, ensemble=(args.model_type=='ensemble'))
            if len(train_data) > 0:
                X_train, y_train, X_val, y_val = predictor.prepare_data(
                    train_data, target_col='Market_State', seq_length=30, test_size=0.2
                )
                model_path = os.path.join("models", f"advanced_market_{args.model_type}_model.pth")
                predictor.train(
                    X_train, y_train, X_val, y_val,
                    batch_size=32, epochs=50, learning_rate=0.001,
                    model_path=model_path
                )
                predictor.save(model_path)
            else:
                model_path = os.path.join(args.output, f"advanced_market_{args.model_type}_model.pth")
                predictor.load(model_path)
        except Exception as e:
            logging.error(f"Advanced model failed: {str(e)}. Falling back to traditional model.")
            args.advanced = False
    
    if not args.advanced:
        from src.prediction_model import MarketPredictor
        predictor = MarketPredictor()
        if len(train_data) > 0:
            if 'ensemble_anomaly' in train_data.columns:
                result = predictor.prepare_features(train_data, 
                                                   anomaly_columns=['ensemble_anomaly', 'ensemble_score'])
            else:
                result = predictor.prepare_features(train_data)
                
            if isinstance(result, (list, tuple)) and len(result) == 4:
                X, y, feature_names, _ = result
            else:
                X, y, feature_names = result
            model, metrics = predictor.train_model(X, y, feature_names)
            predictor.save_model(path=os.path.join(args.output, "market_predictor_model.joblib"))
            predictor.plot_feature_importance(save_path=os.path.join(args.output, "feature_importance.png"))
            predictor.plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=os.path.join(args.output, "confusion_matrix.png"),
                class_names=metrics.get('present_states', None)
            )
        else:
            logging.warning("No training data available. Using default model for predictions.")
            from sklearn.dummy import DummyClassifier
            predictor.model = DummyClassifier(strategy='prior')
            predictor.label_map = {'Bear': 0, 'Static': 1, 'Bull': 2}
            predictor.feature_importance = pd.DataFrame({'Feature': ['PrDec', 'PrInc'], 'Importance': [0.5, 0.5]})
    
    # Step 9: Generate predictions
    logging.info("Generating market predictions")
    
    # Generate predictions using the appropriate model
    if args.advanced and 'AdvancedMarketPredictor' in str(type(predictor)):
        # Prepare test data for the deep learning model
        try:
            # Create sequences from test data for sequential model
            test_sequences = []
            for i in range(len(test_data) - 30):
                window_data = test_data.iloc[i:i+30]
                features = predictor.prepare_data_point(window_data)
                test_sequences.append(features)
            
            if test_sequences:
                import torch
                test_sequences = np.array(test_sequences)
                X_test = torch.tensor(test_sequences, dtype=torch.float32)
                
                # Generate predictions
                pred_labels, pred_probs = predictor.predict(X_test)
                
                # Map predictions back to original data
                test_with_predictions = test_data.copy()
                test_with_predictions['Predicted_Market'] = None
                
                # Add predictions at appropriate locations
                for i, label in enumerate(pred_labels):
                    if i + 30 < len(test_with_predictions):
                        test_with_predictions.iloc[i + 30, test_with_predictions.columns.get_loc('Predicted_Market')] = label
                
                # Forward fill predictions
                test_with_predictions['Predicted_Market'] = test_with_predictions['Predicted_Market'].ffill().bfill()
                
                # Add probability columns
                for i, class_name in enumerate(predictor.classes):
                    col_name = f'{class_name}_Prob'
                    test_with_predictions[col_name] = 0.0  # Default
                    
                    # Fill in probabilities
                    for j, prob in enumerate(pred_probs):
                        if j + 30 < len(test_with_predictions):
                            test_with_predictions.iloc[j + 30, test_with_predictions.columns.get_loc(col_name)] = prob[i]
                    
                    # Forward fill probabilities
                    test_with_predictions[col_name] = test_with_predictions[col_name].ffill().bfill()
            else:
                logging.warning("Failed to create test sequences for advanced model")
                # Fall back to simpler approach
                test_with_predictions = test_data.copy()
                test_with_predictions['Predicted_Market'] = 'Static'  # Default prediction
                test_with_predictions['Bear_Prob'] = 0.1
                test_with_predictions['Static_Prob'] = 0.8
                test_with_predictions['Bull_Prob'] = 0.1
        except Exception as e:
            logging.error(f"Error generating predictions with advanced model: {str(e)}")
            # Fall back to simpler approach
            test_with_predictions = test_data.copy()
            test_with_predictions['Predicted_Market'] = 'Static'  # Default prediction
            test_with_predictions['Bear_Prob'] = 0.1
            test_with_predictions['Static_Prob'] = 0.8
            test_with_predictions['Bull_Prob'] = 0.1
    elif test_data['PrDec'].sum() == 0 and test_data['PrInc'].sum() == 0:
        logging.warning("No meaningful probability data for prediction. Using market trend-based predictions.")
        
        # Add a more basic prediction approach based on market trends
        test_with_predictions = test_data.copy()
        
        # Add some trend indicators
        test_with_predictions['SP500_MA50'] = test_with_predictions['SP500'].rolling(window=50).mean()
        test_with_predictions['SP500_MA200'] = test_with_predictions['SP500'].rolling(window=200).mean()
        
        # Simple trend-based prediction
        test_with_predictions['Predicted_Market'] = 'Static'  # Default
        test_with_predictions.loc[test_with_predictions['SP500'] > test_with_predictions['SP500_MA50'], 'Predicted_Market'] = 'Bull'
        test_with_predictions.loc[test_with_predictions['SP500'] < test_with_predictions['SP500_MA50'], 'Predicted_Market'] = 'Bear'
        
        # Add some fake probabilities for the backtester to use
        test_with_predictions['Bear_Prob'] = 0.0
        test_with_predictions['Static_Prob'] = 0.0
        test_with_predictions['Bull_Prob'] = 0.0
        
        # Set probabilities based on simple trend strength
        for idx, row in test_with_predictions.iterrows():
            if row['Predicted_Market'] == 'Bull':
                test_with_predictions.loc[idx, 'Bull_Prob'] = 0.7
                test_with_predictions.loc[idx, 'Static_Prob'] = 0.2
                test_with_predictions.loc[idx, 'Bear_Prob'] = 0.1
            elif row['Predicted_Market'] == 'Bear':
                test_with_predictions.loc[idx, 'Bear_Prob'] = 0.7
                test_with_predictions.loc[idx, 'Static_Prob'] = 0.2
                test_with_predictions.loc[idx, 'Bull_Prob'] = 0.1
            else:
                test_with_predictions.loc[idx, 'Static_Prob'] = 0.7
                test_with_predictions.loc[idx, 'Bull_Prob'] = 0.15
                test_with_predictions.loc[idx, 'Bear_Prob'] = 0.15
    else:
        test_with_predictions = predictor.generate_predictions(test_data)
    
    # NEW: Ensure anomaly data is transferred to test_with_predictions
    if 'ensemble_anomaly' in anomalies.columns:
        common_dates = test_with_predictions.index.intersection(anomalies.index)
        if not common_dates.empty:
            test_with_predictions['ensemble_anomaly'] = anomalies.loc[common_dates, 'ensemble_anomaly']
            logging.info(f"Added ensemble_anomaly column to predictions data ({len(common_dates)} dates matched)")
            # Display counts to verify
            anomaly_counts = test_with_predictions['ensemble_anomaly'].value_counts()
            logging.info(f"Anomaly distribution: {anomaly_counts.to_dict()}")
        else:
            logging.warning("No common dates between predictions and anomalies - creating dummy anomaly column")
            # Create simple dummy anomaly column
            test_with_predictions['ensemble_anomaly'] = 1  # All normal
    
    # Step 8: Run standard backtests
    logging.info("Running standard investment strategies")
    
    # Check if optimization config is available
    try:
        from config.strategy_optimizations import StrategyOptimizer
        optimizer = StrategyOptimizer()
        use_optimized = True
        logging.info("Optimization parameters loaded from config/strategy_optimizations.py")
        
        # Log some of the optimization parameters to verify
        pred_strategy_params = optimizer.get_parameters('prediction_strategy')
        if pred_strategy_params:
            logging.info(f"Using optimized parameters for prediction strategy:")
            logging.info(f"  - static_allocation: {pred_strategy_params.get('static_allocation', 'N/A')}")
            logging.info(f"  - vol_dampening: {pred_strategy_params.get('vol_dampening', 'N/A')}")
    except ImportError:
        logging.warning("Strategy optimizations not found - using default parameters")
        use_optimized = False
    
    # Initialize backtester with optimization setting
    backtester = StrategyBacktester(test_with_predictions, initial_capital=10000.0, use_optimized=use_optimized)
    
    # Run buy-and-hold strategy
    backtester.run_buy_and_hold()
    
    # Run prediction-based strategy 
    backtester.run_prediction_strategy()
    
    # Run dynamic allocation strategy
    backtester.run_dynamic_allocation_strategy()
    
    # Run combined strategy
    backtester.run_combined_strategy()
    
    # Step 9: Run enhanced backtest strategies
    metrics = None  # Initialize metrics to avoid UnboundLocalError
    
    if args.enhanced:
        logging.info("Running enhanced investment strategies")
        try:
            from src.enhanced_backtester import EnhancedBacktester
            from src.risk_management import TacticalRiskManager, RiskMetrics
            
            # Pass use_optimized to ensure config parameters are used
            enhanced_backtester = EnhancedBacktester(test_with_predictions, use_optimized=use_optimized)
            enhanced_backtester.output_dir = args.output  # Set output directory for plots
            
            # If optimizations are enabled, verify which parameters are being used
            if use_optimized:
                tactical_params = optimizer.get_parameters('tactical_risk_managed')
                regime_params = optimizer.get_parameters('regime_adaptive')
                
                if tactical_params:
                    logging.info(f"Tactical Risk Managed strategy using optimized parameters:")
                    logging.info(f"  - target_vol: {tactical_params.get('target_vol', 'N/A')}")
                    logging.info(f"  - base_bull_allocation: {tactical_params.get('base_bull_allocation', 'N/A')}")
                
                if regime_params:
                    logging.info(f"Regime Adaptive strategy using optimized parameters:")
                    logging.info(f"  - recovery_period: {regime_params.get('recovery_period', 'N/A')}")
                    logging.info(f"  - volatile_allocation: {regime_params.get('volatile_allocation', 'N/A')}")
            
            # First copy over standard strategies for comparison
            enhanced_backtester.results = backtester.results.copy()
            
            # Run tactical risk-managed strategy
            enhanced_backtester.run_tactical_risk_managed_strategy(
                target_vol=0.12, 
                max_leverage=1.0
            )
            
            # Run regime-adaptive strategy
            enhanced_backtester.run_regime_adaptive_strategy()
            
            # Run volatility targeting strategy
            enhanced_backtester.run_volatility_targeting_strategy(
                target_vol=0.10, 
                max_leverage=1.0
            )
            
            # Run market-beating strategy
            enhanced_backtester.run_market_beating_strategy()
            
            # Run combined anomaly-regime strategy if anomaly data is available
            if 'ensemble_anomaly' in test_with_predictions.columns:
                logging.info("Ensemble anomaly column found – running combined anomaly-regime strategy")
                enhanced_backtester.run_combined_anomaly_regime_strategy(
                    prediction_col='Predicted_Market',
                    anomaly_col='ensemble_anomaly'
                )
            else:
                logging.warning("Ensemble anomaly column not found – skipping combined anomaly-regime strategy")
            
            # Run Markov chain strategy if enabled
            if args.markov:
                try:
                    logging.info("Running Markov Chain prediction strategy")
                    from src.markov_strategy import MarkovStrategy
                    
                    markov = MarkovStrategy(test_with_predictions)
                    markov.train_model()
                    
                    markov.plot_transition_heatmap(save_path=os.path.join(args.output, "markov_transition_matrix.png"))
                    markov.plot_steady_state(save_path=os.path.join(args.output, "markov_steady_state.png"))
                    
                    enhanced_backtester.run_markov_chain_strategy(
                        state_column='Market_State',
                        training_window=252
                    )
                    
                    predictions = markov.generate_predictions()
                    accuracy_analysis = pd.DataFrame({
                        'Prediction': predictions['Next_State_Prediction'],
                        'Actual': predictions['Market_State'],
                        'Confidence': predictions['Prediction_Confidence']
                    })
                    
                    accuracy_analysis['Correct'] = accuracy_analysis['Prediction'] == accuracy_analysis['Market_State']
                    overall_accuracy = accuracy_analysis['Correct'].mean()
                    logging.info(f"Markov model prediction accuracy: {overall_accuracy:.2%}")
                    
                except ImportError:
                    logging.error("Markov strategy module not found. Ensure src/markov_strategy.py exists.")
                except Exception as e:
                    logging.error(f"Error running Markov Chain strategy: {str(e)}")
            
            # Generate enhanced performance plots and metrics
            enhanced_backtester.plot_performance(save_path=os.path.join(args.output, "enhanced_strategy_performance.png"))
            enhanced_backtester.plot_regime_allocations(
                strategy_name='regime_adaptive', 
                save_path=os.path.join(args.output, "regime_allocations.png")
            )
            
            enhanced_metrics = enhanced_backtester.summary()
            metrics = enhanced_metrics  # Set metrics to the enhanced metrics
            enhanced_metrics.to_csv(os.path.join(args.output, "enhanced_performance_metrics.csv"))
            
            backtester = enhanced_backtester  # Update backtester reference
            
        except Exception as e:
            logging.error(f"Failed to run enhanced strategies: {str(e)}")
            metrics = backtester.summary()
            metrics.to_csv(os.path.join(args.output, "performance_metrics.csv"))
    else:
        # Generate performance plots and metrics for standard strategies only
        backtester.plot_performance(save_path=os.path.join(args.output, "strategy_performance.png"))
        metrics = backtester.summary()
        metrics.to_csv(os.path.join(args.output, "performance_metrics.csv"))

    # Additional visualizations
    backtester.plot_allocations(strategy_name='prediction_strategy', save_path=os.path.join(args.output, "prediction_allocations.png"))
    backtester.plot_allocations(strategy_name='dynamic_allocation', save_path=os.path.join(args.output, "dynamic_allocations.png"))
    
    # Save predictions
    test_with_predictions.to_csv(os.path.join(args.output, "market_predictions.csv"))
    
    # Step 10: Run specialized strategies if data is available
    if 'ensemble_anomaly' in test_with_predictions.columns:
        logging.info("Running anomaly-responsive investment strategy")
        anomaly_strategy_results = backtester.run_anomaly_aware_strategy(
            anomaly_col='ensemble_anomaly',
            prediction_col='Predicted_Market'
        )
        
        # Generate performance plots for the anomaly-aware strategy
        backtester.plot_allocations(
            strategy_name='anomaly_aware',
            save_path=os.path.join(args.output, "anomaly_strategy_allocations.png")
        )
    
    # Step 11: Print summary of best-performing strategies
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        # Sort strategies by annualized return
        best_strategies = metrics.sort_values('Annualized_Return', ascending=False)
        
        logging.info("\n" + "="*50)
        logging.info("BEST PERFORMING STRATEGIES")
        logging.info("="*50)
        
        # Fix: Convert to a list before slicing
        top_strategies = list(best_strategies.iterrows())[:min(3, len(best_strategies))]
        
        # Print top 3 strategies or all if less than 3
        for i, (strategy, row) in enumerate(top_strategies):
            ann_return = row.get('Annualized_Return', 0) * 100
            max_dd = row.get('Max_Drawdown', 0)
            sharpe = row.get('Sharpe_Ratio', 0)
            
            logging.info(f"{i+1}. {strategy.replace('_', ' ').title()}:")
            logging.info(f"   Return: {ann_return:.2f}% | Max DD: {max_dd:.2f}% | Sharpe: {sharpe:.2f}")
            
        # Compare best strategy to buy and hold
        if 'buy_and_hold' in best_strategies.index and len(best_strategies) > 1:
            best_strategy = best_strategies.index[0]
            if best_strategy != 'buy_and_hold':
                bh_return = best_strategies.loc['buy_and_hold', 'Annualized_Return'] * 100
                best_return = best_strategies.loc[best_strategy, 'Annualized_Return'] * 100
                outperformance = best_return - bh_return
                
                logging.info(f"\nBest strategy outperforms Buy & Hold by {outperformance:.2f}%")
    
    # Final logging before exit
    logging.info("Market prediction and investment pipeline complete.")
    logging.info(f"Results and analysis saved to {args.output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Prediction and Investment Strategy")
    parser.add_argument("--data", type=str, required=True, help="Path to market data file")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results")
    parser.add_argument("--start_date", type=str, help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2022-12-31", help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("--train_start_date", type=str, default="2008-01-01", help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--train_end_date", type=str, help="End date for training data (YYYY-MM-DD)")
    parser.add_argument("--advanced", action="store_true", help="Use advanced PyTorch model")
    parser.add_argument("--model_type", type=str, default="attention", choices=["attention", "tcn", "ensemble"], help="Type of advanced model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--enhanced", action="store_true", help="Run enhanced backtesting strategies")
    parser.add_argument("--markov", action="store_true", help="Use Markov chain prediction strategy")
    
    args = parser.parse_args()
    main(args)