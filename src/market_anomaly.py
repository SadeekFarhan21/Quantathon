import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import logging

class MarketAnomalyDetector:
    """Advanced market anomaly detection using multiple methods"""
    
    def __init__(self, data, contamination=0.05):
        """
        Initialize the anomaly detector
        
        Args:
            data (DataFrame): Market data with price and other features
            contamination (float): Expected proportion of anomalies
        """
        self.data = data
        self.contamination = contamination
        self.results = None
        self.models = {}
        
    def engineer_features(self):
        """
        Engineer features for anomaly detection
        
        Returns:
            DataFrame: Data with engineered features
        """
        df = self.data.copy()
        
        # Calculate returns at different timeframes
        df['Return_1d'] = df['SP500'].pct_change()
        for window in [3, 5, 10, 21]:
            df[f'Return_{window}d'] = df['SP500'].pct_change(window)
            
        # Volatility features
        for window in [5, 10, 21, 63]:
            df[f'Volatility_{window}d'] = df['Return_1d'].rolling(window=window).std()
            
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['Vol_Change'] = df['Volume'].pct_change()
            df['Vol_MA10'] = df['Volume'].rolling(window=10).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_MA10']
        
        # Statistical features
        df['Return_Z'] = (df['Return_1d'] - df['Return_1d'].rolling(window=252).mean()) / \
                         df['Return_1d'].rolling(window=252).std().replace(0, np.finfo(float).eps)
        
        # Technical indicators
        # RSI (Relative Strength Index)
        delta = df['SP500'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bond market features
        if 'BondRate' in df.columns:
            df['Bond_Change'] = df['BondRate'].diff()
            df['Stock_Bond_Ratio'] = df['SP500'] / df['BondRate'].replace(0, np.finfo(float).eps)
            df['Stock_Bond_Change'] = df['Stock_Bond_Ratio'].pct_change()
        
        # Probability-based features
        if 'PrDec' in df.columns and 'PrInc' in df.columns:
            df['Prob_Diff'] = df['PrDec'] - df['PrInc']
            df['Prob_Sum'] = df['PrDec'] + df['PrInc']
            df['Prob_Change'] = df['Prob_Diff'].diff()
            df['Prob_Z'] = (df['Prob_Diff'] - df['Prob_Diff'].rolling(window=63).mean()) / \
                           df['Prob_Diff'].rolling(window=63).std().replace(0, np.finfo(float).eps)
        
        # Fill NaN values with column means
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def detect_anomalies(self, methods=None, feature_sets=None):
        """
        Detect anomalies using multiple methods and feature sets
        
        Args:
            methods (list): List of detection methods to use
            feature_sets (dict): Dictionary of feature sets to test
            
        Returns:
            DataFrame: Data with anomaly scores and flags
        """
        if methods is None:
            methods = ['isolation_forest', 'dbscan', 'statistical']
            
        # Get data with engineered features
        df = self.engineer_features()
        
        # Define feature sets if not provided
        if feature_sets is None:
            feature_sets = {
                'price_movement': ['Return_1d', 'Return_5d', 'Return_21d', 'Volatility_21d'],
                'technical': ['RSI', 'Volatility_10d', 'Return_Z'],
                'probability': ['PrDec', 'PrInc', 'Prob_Diff', 'Prob_Z'] 
                             if 'PrDec' in df.columns else [],
                'combined': ['Return_1d', 'Return_5d', 'Volatility_21d', 'RSI', 
                           'Bond_Change', 'Prob_Diff', 'Prob_Z'] 
                           if 'PrDec' in df.columns else 
                           ['Return_1d', 'Return_5d', 'Volatility_21d', 'RSI', 'Bond_Change']
            }
        
        # Initialize columns for anomaly scores
        for method in methods:
            for feature_set in feature_sets.keys():
                if feature_sets[feature_set]:  # Skip empty feature sets
                    df[f'{method}_{feature_set}_score'] = np.nan
                    df[f'{method}_{feature_set}_anomaly'] = 1  # 1 = normal, -1 = anomaly
        
        # Run each method with each feature set
        for method in methods:
            for feature_set_name, features in feature_sets.items():
                if not features:  # Skip empty feature sets
                    continue
                    
                # Get feature subset and handle missing values
                feature_df = df[features].copy()
                feature_df = feature_df.dropna()
                
                if len(feature_df) <= 1:
                    logging.warning(f"Not enough data for {method} with {feature_set_name} features")
                    continue
                
                # Standardize features
                scaler = StandardScaler()
                X = scaler.fit_transform(feature_df.values)
                
                # Apply dimensionality reduction if needed (for high-dimensional feature sets)
                if X.shape[1] > 5:
                    pca = PCA(n_components=min(5, X.shape[1]))
                    X = pca.fit_transform(X)
                
                # Run detection method
                if method == 'isolation_forest':
                    try:
                        model = IsolationForest(contamination=self.contamination, random_state=42)
                        labels = model.fit_predict(X)
                        scores = model.decision_function(X)
                        
                        # Store model for future use
                        self.models[f'{method}_{feature_set_name}'] = {
                            'model': model,
                            'scaler': scaler,
                            'pca': pca if X.shape[1] > 5 else None
                        }
                        
                        # Add results to dataframe
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_anomaly'] = labels
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_score'] = scores
                    except Exception as e:
                        logging.error(f"Error in {method} with {feature_set_name}: {str(e)}")
                
                elif method == 'dbscan':
                    try:
                        # Estimate epsilon based on data
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=min(10, len(X)))
                        nn.fit(X)
                        distances, _ = nn.kneighbors(X)
                        # Ensure epsilon is always positive by using max
                        eps = max(0.001, np.percentile(distances[:, 1], 95)) if len(distances) > 1 else 0.1
                        
                        # Apply DBSCAN
                        model = DBSCAN(eps=eps, min_samples=min(5, len(X)))
                        labels = model.fit_predict(X)
                        
                        # Convert DBSCAN labels to anomaly format (-1 for anomalies, 1 for normal)
                        # In DBSCAN, -1 already means outlier
                        anomaly_labels = np.ones_like(labels)
                        anomaly_labels[labels == -1] = -1
                        
                        # Calculate scores (distance to nearest core point)
                        from sklearn.metrics.pairwise import euclidean_distances
                        scores = np.ones(len(X))
                        
                        # If any clusters were found
                        if len(np.unique(labels)) > 1:
                            # For each point, find distance to closest core point
                            core_samples = X[labels != -1]
                            if len(core_samples) > 0:
                                for i, point in enumerate(X):
                                    if labels[i] == -1:  # If it's an outlier
                                        # Distance to closest core point
                                        min_dist = np.min(euclidean_distances([point], core_samples))
                                        scores[i] = -min_dist  # Negative distance for consistent scoring
                        
                        # Add results to dataframe
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_anomaly'] = anomaly_labels
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_score'] = scores
                    except Exception as e:
                        logging.error(f"Error in {method} with {feature_set_name}: {str(e)}")
                
                elif method == 'statistical':
                    try:
                        # Statistical outlier detection based on z-scores of multiple features
                        from scipy import stats  # Explicit import here as well
                        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
                        # Handle potential NaNs after z-score calculation
                        z_scores = np.nan_to_num(z_scores, nan=0.0)
                        outliers = np.any(z_scores > 3, axis=1)  # Points with any feature having z-score > 3
                        
                        # Convert to anomaly format
                        anomaly_labels = np.ones(len(X))
                        anomaly_labels[outliers] = -1
                        
                        # Calculate anomaly scores based on maximum z-score
                        max_z = np.max(z_scores, axis=1)
                        scores = -max_z  # Negative because higher z-score means more anomalous
                        
                        # Add results to dataframe
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_anomaly'] = anomaly_labels
                        df.loc[feature_df.index, f'{method}_{feature_set_name}_score'] = scores
                    except Exception as e:
                        logging.error(f"Error in {method} with {feature_set_name}: {str(e)}")
        
        # Create ensemble anomaly score by combining methods
        score_columns = [col for col in df.columns if '_score' in col]
        if score_columns:
            # Normalize scores to [0, 1] range for fair combination
            normalized_scores = pd.DataFrame()
            for col in score_columns:
                scores = df[col].dropna()
                if len(scores) > 0:
                    min_score, max_score = scores.min(), scores.max()
                    if max_score > min_score:
                        normalized_scores[col] = (scores - min_score) / (max_score - min_score)
                    else:
                        normalized_scores[col] = 0.5  # Default value if all scores are the same
            
            # Average normalized scores
            if not normalized_scores.empty:
                df['ensemble_score'] = normalized_scores.mean(axis=1)
                
                # Determine ensemble anomaly labels based on threshold
                ensemble_threshold = np.percentile(df['ensemble_score'].dropna(), 
                                                  self.contamination * 100)
                df['ensemble_anomaly'] = 1
                df.loc[df['ensemble_score'] <= ensemble_threshold, 'ensemble_anomaly'] = -1
        
        self.results = df
        return df
    
    def analyze_anomalies(self, output_dir=None):
        """
        Analyze detected anomalies and their market impact
        
        Args:
            output_dir (str): Directory to save analysis results
            
        Returns:
            dict: Analysis results
        """
        if self.results is None:
            logging.warning("No anomaly detection results found. Running detection first.")
            self.detect_anomalies()
            
        df = self.results
        
        # Get ensemble anomaly dates, or use isolation forest if ensemble not available
        if 'ensemble_anomaly' in df.columns:
            anomaly_dates = df[df['ensemble_anomaly'] == -1].index
        elif 'isolation_forest_combined_anomaly' in df.columns:
            anomaly_dates = df[df['isolation_forest_combined_anomaly'] == -1].index
        else:
            # Use first available anomaly column
            anomaly_cols = [col for col in df.columns if '_anomaly' in col]
            if anomaly_cols:
                anomaly_dates = df[df[anomaly_cols[0]] == -1].index
            else:
                logging.error("No anomaly results found")
                return None
        
        logging.info(f"Analyzing {len(anomaly_dates)} anomalies")
        
        # Analyze market conditions around anomalies
        pre_window = 5  # Days before anomaly
        post_window = 20  # Days after anomaly
        
        anomaly_impacts = []
        
        for date in anomaly_dates:
            try:
                # Get data around the anomaly
                date_idx = df.index.get_loc(date)
                pre_start = max(0, date_idx - pre_window)
                post_end = min(len(df) - 1, date_idx + post_window)
                
                pre_data = df.iloc[pre_start:date_idx]
                post_data = df.iloc[date_idx:post_end+1]
                
                # Calculate impact metrics
                pre_price = df['SP500'].iloc[pre_start]
                anomaly_price = df['SP500'].iloc[date_idx]
                max_post_price = post_data['SP500'].max()
                min_post_price = post_data['SP500'].min()
                
                # Calculate returns
                anomaly_return = df['Return_1d'].iloc[date_idx]
                pre_volatility = pre_data['Return_1d'].std() * np.sqrt(252)  # Annualized
                post_volatility = post_data['Return_1d'].std() * np.sqrt(252)  # Annualized
                post_return = (post_data['SP500'].iloc[-1] / anomaly_price) - 1
                
                # Calculate time to recovery (if price drops)
                recovery_days = None
                if anomaly_return < 0:
                    for i in range(date_idx + 1, post_end + 1):
                        if df['SP500'].iloc[i] >= anomaly_price:
                            recovery_days = (df.index[i] - date).days
                            break
                
                # Collect impact data
                impact = {
                    'Date': date,
                    'SP500': anomaly_price,
                    'Return_1d': anomaly_return,
                    'Pre_Volatility': pre_volatility,
                    'Post_Volatility': post_volatility,
                    'Post_Return': post_return,
                    'Recovery_Days': recovery_days,
                    'Max_Post_Drop': (min_post_price / anomaly_price) - 1,
                    'Max_Post_Gain': (max_post_price / anomaly_price) - 1,
                }
                
                # Add anomaly scores from different methods
                score_columns = [col for col in df.columns if '_score' in col]
                for col in score_columns:
                    if col in df.columns:
                        impact[col] = df[col].iloc[date_idx]
                
                anomaly_impacts.append(impact)
            except Exception as e:
                logging.error(f"Error analyzing anomaly at {date}: {str(e)}")
        
        # Create DataFrame of anomaly impacts
        impact_df = pd.DataFrame(anomaly_impacts)
        
        # Handle empty impact data
        if impact_df.empty:
            logging.warning("No anomaly impacts to analyze")
            return {'anomaly_dates': anomaly_dates, 'impact_data': impact_df, 'summary': {}}
        
        # Calculate summary statistics
        summary = {
            'total_anomalies': len(impact_df),
            'positive_return_anomalies': (impact_df['Return_1d'] > 0).sum(),
            'negative_return_anomalies': (impact_df['Return_1d'] < 0).sum(),
            'avg_anomaly_return': impact_df['Return_1d'].mean(),
            'median_recovery_days': impact_df['Recovery_Days'].median(),
            'post_volatility_ratio': (impact_df['Post_Volatility'] / impact_df['Pre_Volatility']).mean()
        }
        
        # Create visualizations if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save impact data
            if not impact_df.empty:
                impact_df.to_csv(os.path.join(output_dir, 'anomaly_impacts.csv'))
            
            # Create visualization of anomalies on price chart
            self.visualize_anomalies(output_dir)
            
            # Plot impact distribution
            if not impact_df.empty:
                plt.figure(figsize=(15, 10))
                
                # Plot 1: Return distribution on anomaly days
                plt.subplot(2, 2, 1)
                plt.hist(impact_df['Return_1d'], bins=20)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Anomaly Day Returns')
                
                # Plot 2: Recovery time distribution
                plt.subplot(2, 2, 2)
                recovery_days = impact_df['Recovery_Days'].dropna()
                if len(recovery_days) > 0:
                    plt.hist(recovery_days, bins=20)
                    plt.title(f'Recovery Time Distribution (Mean: {recovery_days.mean():.1f} days)')
                else:
                    plt.text(0.5, 0.5, 'No recovery data available', 
                           horizontalalignment='center', verticalalignment='center')
                    plt.title('Recovery Time Distribution')
                
                # Plot 3: Pre vs Post Volatility
                plt.subplot(2, 2, 3)
                plt.scatter(impact_df['Pre_Volatility'], impact_df['Post_Volatility'])
                min_vol = min(impact_df['Pre_Volatility'].min(), impact_df['Post_Volatility'].min())
                max_vol = max(impact_df['Pre_Volatility'].max(), impact_df['Post_Volatility'].max())
                plt.plot([min_vol, max_vol], [min_vol, max_vol], 'r--')
                plt.title('Pre vs Post Anomaly Volatility')
                plt.xlabel('Pre-Anomaly Volatility')
                plt.ylabel('Post-Anomaly Volatility')
                
                # Plot 4: Post-anomaly return distribution
                plt.subplot(2, 2, 4)
                plt.hist(impact_df['Post_Return'], bins=20)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title(f'Post-Anomaly Return ({post_window} days)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'anomaly_impact_analysis.png'))
                plt.close()
        
        return {
            'anomaly_dates': anomaly_dates,
            'impact_data': impact_df,
            'summary': summary
        }
    
    def visualize_anomalies(self, output_dir):
        """
        Create visualizations of detected anomalies
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        df = self.results
        
        # Get primary anomaly column (ensemble or any available)
        if 'ensemble_anomaly' in df.columns:
            primary_anomaly = 'ensemble_anomaly'
        else:
            anomaly_cols = [col for col in df.columns if '_anomaly' in col]
            primary_anomaly = anomaly_cols[0] if anomaly_cols else None
        
        if not primary_anomaly:
            logging.warning("No anomaly results available for visualization")
            return
            
        # Plot price chart with anomalies
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Price with anomalies
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['SP500'], 'b-', label='S&P 500')
        
        anomaly_dates = df[df[primary_anomaly] == -1].index
        if len(anomaly_dates) > 0:
            plt.scatter(anomaly_dates, df.loc[anomaly_dates, 'SP500'], 
                       color='red', marker='o', s=50, label='Anomalies')
        
        plt.title('S&P 500 with Detected Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Returns with anomalies
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['Return_1d'], 'k-', alpha=0.6)
        
        if len(anomaly_dates) > 0:
            plt.scatter(anomaly_dates, df.loc[anomaly_dates, 'Return_1d'], 
                       color='red', marker='o', s=50)
        
        plt.title('Daily Returns with Anomalies')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Anomaly scores
        plt.subplot(3, 1, 3)
        
        # Get main score column
        if 'ensemble_score' in df.columns:
            score_col = 'ensemble_score'
        else:
            score_cols = [col for col in df.columns if '_score' in col]
            score_col = score_cols[0] if score_cols else None
        
        if score_col:
            plt.plot(df.index, df[score_col], 'g-', label='Anomaly Score')
            plt.title(f'Anomaly Scores ({score_col})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomalies_visualization.png'))
        plt.close()
        
        # Create heatmap of anomaly methods agreement
        anomaly_cols = [col for col in df.columns if '_anomaly' in col]
        if len(anomaly_cols) > 1:
            # Create a correlation matrix of anomaly detections
            anomaly_df = pd.DataFrame()
            for col in anomaly_cols:
                anomaly_df[col] = df[col]
            
            # Only create heatmap if we have valid data
            if not anomaly_df.empty and len(anomaly_df) > 1:
                try:
                    plt.figure(figsize=(12, 10))
                    correlation_matrix = anomaly_df.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title('Correlation Between Anomaly Detection Methods')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'anomaly_method_correlation.png'))
                    plt.close()
                except Exception as e:
                    logging.error(f"Error creating method correlation heatmap: {str(e)}")

    def get_anomaly_dates(self):
        """
        Get the dates of detected anomalies
        
        Returns:
            Index: DatetimeIndex of anomaly dates
        """
        if self.results is None:
            logging.warning("No anomaly detection results found")
            return pd.DatetimeIndex([])
        
        # Get ensemble anomaly dates, or use isolation forest if ensemble not available
        if 'ensemble_anomaly' in self.results.columns:
            return self.results[self.results['ensemble_anomaly'] == -1].index
        elif 'isolation_forest_combined_anomaly' in self.results.columns:
            return self.results[self.results['isolation_forest_combined_anomaly'] == -1].index
        else:
            # Use first available anomaly column
            anomaly_cols = [col for col in self.results.columns if '_anomaly' in col]
            if anomaly_cols:
                return self.results[self.results[anomaly_cols[0]] == -1].index
            
        return pd.DatetimeIndex([])
