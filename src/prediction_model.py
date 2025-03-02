import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketPredictor:
    def __init__(self):
        """
        Predict market states using market-based probabilities and other features.
        """
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, data, target_col='Market_State', features=None, forward_periods=63, anomaly_columns=None):
        """
        Prepare features and target for the prediction model with enhanced features.
        
        Args:
            data (DataFrame): Market data with classified states
            target_col (str): Column name for market state
            features (list): List of base feature columns
            forward_periods (int): Number of periods ahead to predict
            anomaly_columns (list): List of anomaly columns to include as features
            
        Returns:
            tuple: X (features), y (target), and feature names
        """
        logging.info("Preparing features for prediction model")
        
        # If no features specified, use default
        if features is None:
            features = ['PrDec', 'PrInc']
        
        # Create a copy of the data
        df = data.copy()
        
        # Create a future market state target
        df['Future_Market'] = df[target_col].shift(-forward_periods)
        
        # Basic engineered features
        df['PrDec_PrInc_Ratio'] = df['PrDec'] / df['PrInc'].replace(0, 0.001)  # Avoid division by zero
        df['PrDiff'] = df['PrDec'] - df['PrInc']
        
        # Add rolling statistics on the probabilities
        for window in [5, 10, 20]:
            df[f'PrDec_MA{window}'] = df['PrDec'].rolling(window=window).mean()
            df[f'PrInc_MA{window}'] = df['PrInc'].rolling(window=window).mean()
            df[f'PrDiff_MA{window}'] = df['PrDiff'].rolling(window=window).mean()
            
        # Add recent market trends
        for window in [5, 20, 60]:
            df[f'SP500_Return_{window}d'] = df['SP500'].pct_change(window)
            
        # Add market volatility
        df['Volatility_20d'] = df['SP500'].pct_change().rolling(window=20).std()
        
        # Add bond rate features
        df['BondRate_Change_5d'] = df['BondRate'].diff(5)
        df['SP500_BondRate_Ratio'] = df['SP500'] / df['BondRate'].replace(0, 0.001)
        
        # ENHANCED FEATURES
        
        # 1. Trend strength indicators
        df['PrDec_Trend'] = df['PrDec'].pct_change(20).rolling(window=5).mean()
        df['PrInc_Trend'] = df['PrInc'].pct_change(20).rolling(window=5).mean()
        
        # 2. Volatility ratio
        df['Volatility_Ratio'] = df['Volatility_20d'] / df['Volatility_20d'].rolling(window=60).mean()
        
        # 3. Probability extremes - detect unusual values
        df['PrDec_Extreme'] = (df['PrDec'] - df['PrDec'].rolling(90).mean()) / df['PrDec'].rolling(90).std()
        df['PrInc_Extreme'] = (df['PrInc'] - df['PrInc'].rolling(90).mean()) / df['PrInc'].rolling(90).std()
        
        # 4. Market regime detection
        df['Bull_Regime'] = ((df['SP500'].pct_change(60) > 0) & (df['SP500'].pct_change(20) > 0)).astype(int)
        df['Bear_Regime'] = ((df['SP500'].pct_change(60) < 0) & (df['SP500'].pct_change(20) < 0)).astype(int)
        
        # 5. Bond yield dynamics
        df['Yield_Trend'] = df['BondRate'].pct_change(20).rolling(window=10).mean()
        
        # Create the complete feature list
        extended_features = features + [
            'PrDec_PrInc_Ratio', 'PrDiff', 
            'PrDec_MA5', 'PrInc_MA5', 'PrDiff_MA5',
            'PrDec_MA20', 'PrInc_MA20', 'PrDiff_MA20',
            'SP500_Return_5d', 'SP500_Return_20d', 'SP500_Return_60d',
            'Volatility_20d', 'BondRate_Change_5d', 'SP500_BondRate_Ratio',
            'PrDec_Trend', 'PrInc_Trend', 'Volatility_Ratio',
            'PrDec_Extreme', 'PrInc_Extreme',
            'Bull_Regime', 'Bear_Regime', 'Yield_Trend'
        ]
        
        # Add anomaly columns if provided
        if anomaly_columns:
            for col in anomaly_columns:
                if col in df.columns:
                    extended_features.append(col)
                    print(f"Adding anomaly feature: {col}")
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Extract features and target
        X = df[extended_features].values
        y = df['Future_Market'].values
        
        return X, y, extended_features
        
    def train_model(self, X, y, feature_names, test_size=0.2, random_state=42):
        """
        Train an advanced model to predict market states with improved handling of class imbalance
        """
        logging.info("Training market prediction model with enhanced regularization and balance handling")
        
        # Convert market states to numeric with a consistent mapping
        self.label_map = {'Bear': 0, 'Static': 1, 'Bull': 2}
        y_numeric = np.array([self.label_map.get(state, 1) for state in y])
        
        # Check class distribution
        unique_classes, counts = np.unique(y_numeric, return_counts=True)
        class_dist = dict(zip([['Bear', 'Static', 'Bull'][c] for c in unique_classes], counts))
        logging.info(f"Class distribution in training data: {class_dist}")
        
        # If we have severe class imbalance, address it
        if len(unique_classes) < 3:
            logging.warning(f"Only found {len(unique_classes)} classes in training data!")
            
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_numeric)
        if len(classes) > 1:  # Make sure we have at least 2 classes
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_numeric)
            class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
            logging.info(f"Using class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
            logging.warning("Only one class found in training data, class weighting disabled")
        
        # Split data with stratification if possible
        from sklearn.model_selection import train_test_split
        if len(unique_classes) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_numeric, test_size=test_size, 
                shuffle=True,  # Enable shuffling for better distribution
                stratify=y_numeric,  # Stratify to maintain class distributions
                random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_numeric, test_size=test_size, shuffle=True, random_state=random_state
            )
        
        # Check if stratification worked
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        logging.info(f"Classes in training set: {train_classes}, counts: {train_counts}")
        
        # Use a more robust model for imbalanced data
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Custom parameters based on class balance
        if len(unique_classes) == 1:
            # If only one class, use a simple model
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=unique_classes[0])
            model.fit(X_train, y_train)
            logging.warning("Using DummyClassifier because only one class found")
        else:
            # Define a smaller, more focused parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4],
                'subsample': [0.8],
                'min_samples_split': [5]
            }
            
            # Base estimator with class weights
            base_model = GradientBoostingClassifier(
                random_state=random_state,
                validation_fraction=0.2,
                n_iter_no_change=10,
                tol=0.001
            )
            
            # Custom TimeSeriesSplit to ensure each fold has all classes
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits to ensure each has enough samples
            
            # GridSearchCV with handling for failed fits
            from sklearn.model_selection import GridSearchCV
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=tscv, 
                scoring='accuracy', 
                n_jobs=-1, verbose=1,
                error_score=0  # Return score=0 for failed fits instead of raising error
            )
            
            try:
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters: {grid_search.best_params_}")
            except Exception as e:
                logging.error(f"GridSearchCV failed: {str(e)}")
                # Fallback to basic model with safe parameters
                best_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
                best_model.fit(X_train, y_train)
        
        # Evaluate the model safely
        try:
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get present classes with safe handling
            present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            states = ['Bear', 'Static', 'Bull']
            present_states = [states[i] for i in present_classes if i < len(states)]
            
            logging.info(f"Classes present in test data: {present_states}")
            
            # Confusion matrix with only present classes
            conf_matrix = confusion_matrix(y_test, y_pred, labels=present_classes)
            
            # Classification report with zero_division parameter
            report = classification_report(
                y_test, y_pred, 
                labels=present_classes,
                target_names=present_states, 
                output_dict=True,
                zero_division=0  # Handle division by zero
            )
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            accuracy = 0
            conf_matrix = np.array([[0]])
            present_classes = [1]
            present_states = ['Static']
            report = {'Static': {'precision': 0, 'recall': 0, 'f1-score': 0}}
        
        # Store the trained model
        self.model = best_model
        
        # Add feature permutation importance for better feature selection
        from sklearn.inspection import permutation_importance
        
        perm_importance = permutation_importance(
            best_model, X_test, y_test, 
            n_repeats=10, random_state=random_state
        )
        
        # Create feature importance DataFrame with both methods
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_,
            'Permutation_Importance': perm_importance.importances_mean
        }).sort_values(by='Permutation_Importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # Log model performance
        logging.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'feature_importance': self.feature_importance,
            'present_classes': present_classes,
            'present_states': present_states
        }
        
        return best_model, metrics
    
    def generate_predictions(self, data):
        """
        Generate predictions with probability estimates for the given data.
        
        Args:
            data (DataFrame): Data with market states and probability features
            
        Returns:
            DataFrame: Data with predictions and probability estimates added
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train or load a model first.")
        
        # Create a copy of the data
        df = data.copy()
        
        # Prepare features (same as in prepare_features method)
        df['PrDec_PrInc_Ratio'] = df['PrDec'] / df['PrInc'].replace(0, 0.001)
        df['PrDiff'] = df['PrDec'] - df['PrInc']
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            df[f'PrDec_MA{window}'] = df['PrDec'].rolling(window=window).mean()
            df[f'PrInc_MA{window}'] = df['PrInc'].rolling(window=window).mean()
            df[f'PrDiff_MA{window}'] = df['PrDiff'].rolling(window=window).mean()
            
        # Add recent market trends
        for window in [5, 20, 60]:
            df[f'SP500_Return_{window}d'] = df['SP500'].pct_change(window)
            
        # Add market volatility
        df['Volatility_20d'] = df['SP500'].pct_change().rolling(window=20).std()
        
        # Add bond rate features
        df['BondRate_Change_5d'] = df['BondRate'].diff(5)
        df['SP500_BondRate_Ratio'] = df['SP500'] / df['BondRate'].replace(0, 0.001)
        
        # Enhanced features
        df['PrDec_Trend'] = df['PrDec'].pct_change(20).rolling(window=5).mean()
        df['PrInc_Trend'] = df['PrInc'].pct_change(20).rolling(window=5).mean()
        df['Volatility_Ratio'] = df['Volatility_20d'] / df['Volatility_20d'].rolling(window=60).mean()
        df['PrDec_Extreme'] = (df['PrDec'] - df['PrDec'].rolling(90).mean()) / df['PrDec'].rolling(90).std()
        df['PrInc_Extreme'] = (df['PrInc'] - df['PrInc'].rolling(90).mean()) / df['PrInc'].rolling(90).std()
        df['Bull_Regime'] = ((df['SP500'].pct_change(60) > 0) & (df['SP500'].pct_change(20) > 0)).astype(int)
        df['Bear_Regime'] = ((df['SP500'].pct_change(60) < 0) & (df['SP500'].pct_change(20) < 0)).astype(int)
        df['Yield_Trend'] = df['BondRate'].pct_change(20).rolling(window=10).mean()
        
        # Extract features for prediction
        features = [
            'PrDec', 'PrInc', 
            'PrDec_PrInc_Ratio', 'PrDiff', 
            'PrDec_MA5', 'PrInc_MA5', 'PrDiff_MA5',
            'PrDec_MA20', 'PrInc_MA20', 'PrDiff_MA20',
            'SP500_Return_5d', 'SP500_Return_20d', 'SP500_Return_60d',
            'Volatility_20d', 'BondRate_Change_5d', 'SP500_BondRate_Ratio',
            'PrDec_Trend', 'PrInc_Trend', 'Volatility_Ratio',
            'PrDec_Extreme', 'PrInc_Extreme',
            'Bull_Regime', 'Bear_Regime', 'Yield_Trend'
        ]
        
        # Fill missing values for prediction
        for col in features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Remove rows with missing features
        df_for_pred = df.dropna(subset=features)
        
        # Make predictions with probabilities
        numeric_predictions = self.model.predict(df_for_pred[features].values)
        class_probabilities = self.model.predict_proba(df_for_pred[features].values)
        
        # States mapping
        states = ['Bear', 'Static', 'Bull']
        
        # Convert numeric predictions to string labels
        predictions = [states[p] for p in numeric_predictions]
        
        # Add predictions to dataframe
        df_for_pred['Predicted_Market'] = predictions  # Explicitly set as object/string type
        
        # Add probabilities to dataframe
        if class_probabilities.shape[1] >= 3:  # If we have all three classes
            df_for_pred['Bear_Prob'] = class_probabilities[:, 0]
            df_for_pred['Static_Prob'] = class_probabilities[:, 1]
            df_for_pred['Bull_Prob'] = class_probabilities[:, 2]
        elif class_probabilities.shape[1] == 2:  # If we only have two classes
            present_classes = sorted(np.unique(numeric_predictions))
            for i, class_idx in enumerate(present_classes):
                class_name = states[class_idx]
                df_for_pred[f'{class_name}_Prob'] = class_probabilities[:, i]
        
        # Merge predictions back to original dataframe - ensure correct dtype handling
        df['Predicted_Market'] = None  # Initialize with None to ensure object dtype
        df.loc[df_for_pred.index, 'Predicted_Market'] = df_for_pred['Predicted_Market']
        
        # Merge probability columns
        prob_cols = [col for col in df_for_pred.columns if col.endswith('_Prob')]
        for col in prob_cols:
            df[col] = np.nan
            df.loc[df_for_pred.index, col] = df_for_pred[col]
            df[col] = df[col].ffill()
        
        # Forward fill predictions to cover all dates
        df['Predicted_Market'] = df['Predicted_Market'].ffill()
        
        return df
    
    def save_model(self, path='results/market_predictor.joblib'):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            logging.error("No trained model to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"Model saved to {path}")
        
    def load_model(self, path='results/market_predictor.joblib'):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        if not os.path.exists(path):
            logging.error(f"Model file not found: {path}")
            return False
            
        self.model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        return True
        
    def predict(self, features):
        """
        Predict market state using the trained model.
        
        Args:
            features (array): Feature vector
            
        Returns:
            str: Predicted market state
        """
        if self.model is None:
            logging.error("No trained model available. Train or load a model first.")
            return None
            
        # Map numeric predictions back to market states
        states = ['Bear', 'Static', 'Bull']
        prediction = self.model.predict([features])[0]
        
        return states[prediction]
        
    def plot_feature_importance(self, save_path='results/feature_importance.png'):
        """
        Plot feature importance from the trained model.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.feature_importance is None:
            logging.error("No feature importance available. Train a model first.")
            return
            
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=self.feature_importance.head(15)
        )
        plt.title('Top 15 Feature Importance for Market Prediction')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Feature importance plot saved to {save_path}")
        
    def plot_confusion_matrix(self, conf_matrix, save_path='results/confusion_matrix.png', 
                              class_names=None):
        """
        Plot confusion matrix for the model evaluation.
        
        Args:
            conf_matrix (array): Confusion matrix
            save_path (str): Path to save the plot
            class_names (list): Names of classes to use in the plot
        """
        if class_names is None:
            class_names = ['Bear', 'Static', 'Bull']
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion matrix plot saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    from data_loader_market import MarketDataLoader
    from market_classifier import MarketClassifier
    
    # Load and prepare data
    loader = MarketDataLoader()
    merged_data = loader.merge_data()
    
    # Classify markets
    classifier = MarketClassifier(merged_data)
    market_states = classifier.classify_markets()
    
    # Create a combined dataset
    data = market_states.join(merged_data[['PrDec', 'PrInc']])
    
    # Create and train the prediction model
    predictor = MarketPredictor()
    X, y, feature_names = predictor.prepare_features(data)
    model, metrics = predictor.train_model(X, y, feature_names)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(metrics['confusion_matrix'], class_names=metrics['present_states'])
    
    # Save the model
    predictor.save_model()
