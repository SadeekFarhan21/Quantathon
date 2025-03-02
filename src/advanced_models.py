import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

class AttentionLSTM(nn.Module):
    """LSTM model with self-attention mechanism for better feature importance learning"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize weights with a better method
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        # Bidirectional LSTM for capturing past and future patterns
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Multiple dense layers with stronger regularization
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.out = nn.Linear(64, num_classes)
        
        # Apply weight initialization
        self.apply(init_weights)
        
    def forward(self, x):
        # Check for NaNs in input
        if torch.isnan(x).any():
            # Replace NaNs with zeros
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        try:
            # Forward propagate LSTM
            lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: batch_size, seq_len, hidden_size*2
            
            # Apply attention
            attention_scores = self.attention(lstm_out).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Create context vector as weighted sum of outputs
            context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
            
            # Dense layers with residual connections
            out = self.fc1(context)
            out = self.bn1(out)
            out = torch.relu(out)
            out = self.dropout1(out)
            
            out = self.fc2(out)
            out = self.bn2(out)
            out = torch.relu(out)
            out = self.dropout2(out)
            
            out = self.out(out)
            
            # Check for NaNs in output
            if torch.isnan(out).any():
                # Return zeros with a small positive bias if there are NaNs
                logging.warning("NaNs detected in model output. Using fallback output.")
                return torch.ones(batch_size, self.num_classes) / self.num_classes
                
            return out
            
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            # Return a fallback output in case of error
            return torch.ones(batch_size, self.num_classes) / self.num_classes

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for market prediction"""
    def __init__(self, input_size, seq_length, num_classes, num_channels=[64, 128, 64], kernel_size=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        
        # First convolution layer directly from input
        layers += [
            nn.Conv1d(input_size, num_channels[0], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]
        
        # Additional convolutional blocks
        for i in range(len(num_channels)-1):
            layers += [
                nn.Conv1d(num_channels[i], num_channels[i+1], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ]
        
        # Global average pooling to handle variable sequence lengths
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense classifier
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # Reshape input to (batch_size, features, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply TCN layers
        x = self.tcn(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classifier
        return self.fc(x)

class EnsembleModel:
    """Ensemble model combining multiple approaches for more robust predictions"""
    def __init__(self, models_list):
        self.models = models_list
        
    def predict(self, X):
        # Get predictions from each model
        all_preds = []
        all_probs = []
        
        for model in self.models:
            preds, probs = model.predict(X)
            all_preds.append(preds)
            all_probs.append(probs)
            
        # Convert to arrays for easier processing
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Ensemble predictions (majority vote)
        # Take mode along the first axis (models axis)
        ensemble_preds = mode(all_preds, axis=0)[0]
        
        # Ensemble probabilities (average)
        ensemble_probs = np.mean(all_probs, axis=0)
        
        return ensemble_preds, ensemble_probs
        
class AdvancedMarketPredictor:
    def __init__(self, model_type='attention', ensemble=False):
        """
        Advanced deep learning models for market prediction with stronger regularization
        
        Args:
            model_type (str): Model type ('attention', 'tcn', or 'transformer')
            ensemble (bool): Whether to use ensemble learning
        """
        self.model_type = model_type
        self.ensemble = ensemble
        self.models = []
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_importance = None
        
    def prepare_data(self, df, target_col='Market_State', seq_length=30, test_size=0.2):
        """Prepare data with more extensive feature engineering"""
        # Define base features
        base_features = ['PrDec', 'PrInc']
        
        # Create a copy for feature engineering
        df = df.copy()
        
        # Basic engineered features
        df['PrDec_PrInc_Ratio'] = df['PrDec'] / df['PrInc'].replace(0, 0.001)
        df['PrDiff'] = df['PrDec'] - df['PrInc']
        
        # Technical indicators
        # 1. Rolling statistics
        for window in [5, 10, 20, 30, 60]:
            # Probability features
            df[f'PrDec_MA{window}'] = df['PrDec'].rolling(window=window).mean()
            df[f'PrInc_MA{window}'] = df['PrInc'].rolling(window=window).mean()
            
            # Price based features if available
            if 'SP500' in df.columns:
                df[f'SP500_MA{window}'] = df['SP500'].rolling(window=window).mean()
                df[f'SP500_Return_{window}d'] = df['SP500'].pct_change(window)
                df[f'SP500_Std_{window}d'] = df['SP500'].pct_change().rolling(window).std()
                
                # Add crossover signals
                if window in [20, 50]:
                    df[f'SP500_Above_MA{window}'] = (df['SP500'] > df[f'SP500_MA{window}']).astype(int)
        
        # 2. Momentum indicators
        if 'SP500' in df.columns:
            df['RSI_14'] = self._calculate_rsi(df['SP500'], 14)
            
        # 3. Volatility indicators
        df['Prob_Volatility'] = (df['PrDec'] - df['PrDec'].shift(1)).rolling(10).std() + \
                               (df['PrInc'] - df['PrInc'].shift(1)).rolling(10).std()
        
        # 4. Rate of change features
        for col in ['PrDec', 'PrInc']:
            for period in [1, 5, 10]:
                df[f'{col}_ROC_{period}'] = (df[col] - df[col].shift(period)) / df[col].shift(period)
                
        # 5. Cross-feature relationships
        if 'BondRate' in df.columns:
            df['SP500_Bond_Ratio'] = df['SP500'] / df['BondRate'].replace(0, 0.001)
            df['BondRate_Change_5d'] = df['BondRate'].diff(5)
            
        # Add more features if specific columns exist
        possible_features = ['SP500', 'BondRate', 'Is_Bear', 'Is_Bull', 'Drawdown']
        additional_features = [col for col in possible_features if col in df.columns]
        
        # Combine all features
        features_to_use = list(base_features)
        
        # Add engineered features
        for col in df.columns:
            if (col.startswith('PrDec_') or 
                col.startswith('PrInc_') or 
                col.startswith('SP500_') or 
                col.startswith('RSI_') or
                col.endswith('_ROC_5') or  # Use only some ROC features to avoid bloat
                col in ['PrDec_PrInc_Ratio', 'PrDiff', 'Prob_Volatility']):
                features_to_use.append(col)
        
        # Add the basic market features
        features_to_use.extend(additional_features)
        
        # Remove duplicates while preserving order
        self.feature_columns = list(dict.fromkeys(features_to_use))
        
        # Drop NaN values and impute remaining
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For columns with high NaN count, forward fill then backward fill
        for col in self.feature_columns:
            if col in df.columns:
                nan_pct = df[col].isna().mean()
                if nan_pct > 0:
                    df[col] = df[col].ffill().bfill()
                    
        # Scale features
        try:
            features = df[self.feature_columns].values
            self.feature_scaler.fit(features)
            scaled_features = self.feature_scaler.transform(features)
        except KeyError as e:
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            raise ValueError(f"Missing columns in dataset: {missing_cols}. Error: {str(e)}")
            
        # Encode target
        labels = df[target_col].values
        self.label_encoder.fit(labels)
        self.classes = self.label_encoder.classes_
        encoded_labels = self.label_encoder.transform(labels)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - seq_length):
            X.append(scaled_features[i:i+seq_length])
            y.append(encoded_labels[i+seq_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # Train/validation split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logging.info(f"Using {len(self.feature_columns)} features: {self.feature_columns[:10]}...")
        logging.info(f"Prepared {len(X_train)} training sequences, {len(X_val)} validation sequences")
        logging.info(f"Target classes: {self.classes}")
        
        return X_train, y_train, X_val, y_val
        
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate EMAs
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def create_model(self, input_size, seq_length, num_classes):
        """Create advanced model based on selected type"""
        if self.model_type == 'attention':
            model = AttentionLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.4  # Higher dropout for regularization
            )
        elif self.model_type == 'tcn':
            model = TemporalConvNet(
                input_size=input_size,
                seq_length=seq_length,
                num_classes=num_classes
            )
        else:  # Default to transformer
            from src.pytorch_models import TransformerModel
            model = TransformerModel(
                input_size=input_size,
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=512,
                num_classes=num_classes
            )
            
        return model.to(self.device)
    
    # Improve the train method with better early stopping
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100,
             learning_rate=0.001, patience=10, model_path="results/advanced_market_model.pth"):
        """Train the model with stricter early stopping and regularization to prevent overfitting"""
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Dimensions
        input_size = X_train.shape[2]  # Number of features
        seq_length = X_train.shape[1]  # Sequence length
        num_classes = len(np.unique(y_train))
        
        # Create models
        models_to_train = []
        if self.ensemble:
            # Create multiple models with different architectures
            models_to_train = [
                self.create_model(input_size, seq_length, num_classes) for _ in range(3)
            ]
        else:
            # Single model
            models_to_train = [self.create_model(input_size, seq_length, num_classes)]
            
        # Data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        for model_idx, model in enumerate(models_to_train):
            model_name = f"{self.model_type}_{model_idx}" if self.ensemble else self.model_type
            
            # Set up optimizer with lower learning rate and higher weight decay
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate * 0.7,  # Lower learning rate to prevent overfitting
                weight_decay=0.02  # Higher L2 regularization
            )
            
            # Scheduler - more aggressive learning rate reduction
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6, verbose=True
            )
            
            # Loss function with class weighting to handle imbalance
            class_weights = torch.ones(num_classes, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Training loop with IMPROVED early stopping
            best_val_loss = float('inf')
            best_val_acc = 0.0
            early_stop_counter = 0
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            
            logging.info(f"Training {model_name} model with improved early stopping...")
            
            best_model_state = None  # Store best model state in memory
            
            # Track the minimum validation loss improvement required to reset early stopping
            min_improvement = 0.001  # Require at least 0.1% improvement
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Stronger gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                    
                train_loss = train_loss / train_total
                train_acc = train_correct / train_total
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Validation with more detailed monitoring
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                class_correct = [0] * num_classes
                class_total = [0] * num_classes
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        
                        # Track per-class accuracy
                        for c in range(num_classes):
                            class_mask = targets == c
                            class_correct[c] += (predicted[class_mask] == c).sum().item()
                            class_total[c] += class_mask.sum().item()
                            
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Check for overfitting: if training acc is much higher than val acc
                overfit_gap = train_acc - val_acc
                overfitting = overfit_gap > 0.15  # More than 15% gap indicates overfitting
                
                # Detailed per-class accuracy
                class_acc_str = " | ".join([f"C{i}: {(c/t)*100:.1f}%" if t > 0 else "N/A" for i, (c, t) in enumerate(zip(class_correct, class_total))])
                
                # Early stopping check - more nuanced
                if val_loss < (best_val_loss - min_improvement):
                    # Clear improvement
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    
                    # Save best model state in memory
                    best_model_state = model.state_dict().copy()
                    logging.info(f"[{model_name}] Epoch {epoch+1}: New best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
                    
                elif val_acc > (best_val_acc + min_improvement*10):  # 1% accuracy improvement
                    # Accuracy improved even if loss didn't
                    best_val_acc = val_acc
                    early_stop_counter = max(0, early_stop_counter - 1)  # Reduce counter but don't reset completely
                    logging.info(f"[{model_name}] Epoch {epoch+1}: Accuracy improved but not loss")
                    
                    # Save if we don't have a best state yet
                    if best_model_state is None:
                        best_model_state = model.state_dict().copy()
                else:
                    early_stop_counter += 1
                    
                    # If high accuracy gap, increment counter more aggressively
                    if overfitting:
                        early_stop_counter += 1
                        logging.warning(f"[{model_name}] Overfitting detected: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, gap={overfit_gap:.4f}")
                    
                # Log progress with overfitting indicator
                if (epoch + 1) % 5 == 0 or epoch + 1 == epochs or early_stop_counter >= patience:
                    logging.info(
                        f"[{model_name}] Epoch {epoch+1}/{epochs}: "
                        f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} "
                        f"| {class_acc_str}"
                        f"| Early stop={early_stop_counter}/{patience}"
                        f"{'  ⚠️ OVERFITTING' if overfitting else ''}"
                    )
                    
                # Early stopping with lower patience for clear overfitting cases
                if early_stop_counter >= patience or (overfitting and early_stop_counter >= patience // 2):
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
            # Save the best model
            model_save_path = model_path.replace('.pth', f'_{model_name}.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            if best_model_state is not None:
                # Load the best model state back into the model
                model.load_state_dict(best_model_state)
                # Then save it
                torch.save(best_model_state, model_save_path)
            else:
                # If no best state (unlikely), save current state
                torch.save(model.state_dict(), model_save_path)
                
            self.models.append(model)
            
            # Plot training curves with overfitting visualization
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.title(f'{model_name} - Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train')
            plt.plot(val_accs, label='Validation')
            plt.fill_between(range(len(train_accs)), val_accs, train_accs, 
                             where=(np.array(train_accs) > np.array(val_accs)),
                             color='red', alpha=0.3, label='Overfitting')
            plt.title(f'{model_name} - Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plot_dir = os.path.dirname(model_path)
            plt.savefig(os.path.join(plot_dir, f"{model_name}_training_curves.png"))
            plt.close()
            
        logging.info("All models trained successfully")
        
    def predict(self, X):
        """Generate predictions with the trained model(s)"""
        if not self.models:
            raise ValueError("No trained models available")
            
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
        # Ensemble predictions
        if self.ensemble and len(self.models) > 1:
            all_preds = []
            all_probs = []
            
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(probabilities, 1)
                    all_preds.append(predictions.cpu().numpy())
                    all_probs.append(probabilities.cpu().numpy())
                    
            # Convert to arrays
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            
            # Ensemble by voting
            final_preds = []
            for i in range(all_preds.shape[1]):
                # Count occurrences of each class
                values, counts = np.unique(all_preds[:, i], return_counts=True)
                final_preds.append(values[np.argmax(counts)])
                
            final_preds = np.array(final_preds)
            
            # Average probabilities
            final_probs = np.mean(all_probs, axis=0)
            
        else:
            # Single model prediction
            model = self.models[0]
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                final_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predictions = torch.max(outputs, 1)
                final_preds = predictions.cpu().numpy()
                
        # Convert numeric predictions to original labels
        pred_labels = self.label_encoder.inverse_transform(final_preds)
        
        return pred_labels, final_probs
        
    def analyze_feature_importance(self, X_val, y_val, feature_names, save_path='results/feature_analysis.png'):
        """Analyze feature importance using permutation importance"""
        if not self.models:
            raise ValueError("No trained models available")
            
        # Convert to torch tensor
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        # Reference accuracy
        pred_labels, _ = self.predict(X_tensor)
        baseline_accuracy = (pred_labels == self.label_encoder.inverse_transform(y_val)).mean()
        
        # Permutation importance
        importances = []
        for i, feature in enumerate(feature_names):
            # Copy the data
            X_permuted = X_val.copy()
            
            # Permute this feature
            np.random.shuffle(X_permuted[:, :, i])
            
            # Get predictions
            X_tensor_perm = torch.tensor(X_permuted, dtype=torch.float32).to(self.device)
            pred_labels_perm, _ = self.predict(X_tensor_perm)
            
            # Calculate accuracy drop
            accuracy_perm = (pred_labels_perm == self.label_encoder.inverse_transform(y_val)).mean()
            importance = baseline_accuracy - accuracy_perm
            
            importances.append(importance)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('Feature Importance (Top 15)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return importance_df
        
    def save(self, path):
        """Save model and metadata"""
        if not self.models:
            raise ValueError("No models to save")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save each model
        model_paths = []
        for i, model in enumerate(self.models):
            model_name = f"{self.model_type}_{i}" if self.ensemble else self.model_type
            # Fix: Generate model path for each model instead of using undefined variable
            current_model_path = path.replace('.pth', f'_{model_name}.pth') if self.ensemble else path
            torch.save(model.state_dict(), current_model_path)
            model_paths.append(current_model_path)
            
        # Save metadata
        import pickle
        meta_path = path.replace('.pth', '_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'classes': self.classes,
                'model_type': self.model_type,
                'ensemble': self.ensemble,
                'model_paths': model_paths
            }, f)
            
        logging.info(f"Models saved to {path} and metadata saved to {meta_path}")
        
    def load(self, path):
        """Load model and metadata"""
        meta_path = path.replace('.pth', '_meta.pkl')
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file not found: {meta_path}")
            
        # Load metadata
        import pickle
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.feature_scaler = metadata['feature_scaler']
        self.label_encoder = metadata['label_encoder']
        self.feature_columns = metadata['feature_columns']
        self.classes = metadata['classes']
        self.model_type = metadata['model_type']
        self.ensemble = metadata['ensemble']
        
        # Load models
        input_size = len(self.feature_columns)
        seq_length = 30  # Default
        if self.classes is not None and hasattr(self.classes, '__len__'):
            num_classes = len(self.classes)
        else:
            num_classes = 3
        
        model_paths = metadata.get('model_paths', [path])
        self.models = []
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = self.create_model(input_size, seq_length, num_classes)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.models.append(model)
                
        if not self.models:
            raise ValueError("Failed to load any models")
            
        logging.info(f"Loaded {len(self.models)} models")
        return True
        
    def prepare_data_point(self, df):
        """
        Prepare a single window of data for prediction
        
        Args:
            df: DataFrame containing a window of data
            
        Returns:
            array: Processed features
        """
        # Create the same features used during training
        try:
            # Basic features
            df_copy = df.copy()
            
            # Make sure all required columns exist
            for col in ['PrDec', 'PrInc', 'SP500', 'BondRate']:
                if col not in df_copy.columns:
                    if col in ['PrDec', 'PrInc']:
                        df_copy[col] = 0.0  # Default values for missing probability columns
                    else:
                        raise ValueError(f"Required column {col} missing from data")
            
            # Create engineered features
            df_copy['PrDec_PrInc_Ratio'] = df_copy['PrDec'] / df_copy['PrInc'].replace(0, 0.001)
            df_copy['PrDiff'] = df_copy['PrDec'] - df_copy['PrInc']
            
            # Technical indicators (subset for this point-wise processing)
            if 'SP500' in df_copy.columns:
                df_copy['RSI_14'] = self._calculate_rsi(df_copy['SP500'], 14)
                
                # Create some simple MAs
                for window in [5, 10, 20]:
                    df_copy[f'SP500_MA{window}'] = df_copy['SP500'].rolling(window=window).mean()
                    df_copy[f'SP500_Return_{window}d'] = df_copy['SP500'].pct_change(window)
            
            # Probability volatility
            df_copy['Prob_Volatility'] = (df_copy['PrDec'] - df_copy['PrDec'].shift(1)).rolling(10).std() + \
                                         (df_copy['PrInc'] - df_copy['PrInc'].shift(1)).rolling(10).std()
            
            # Fill NaN values - using ffill() and bfill() methods instead of fillna(method=)
            df_copy = df_copy.ffill().bfill().fillna(0)
            
            # Extract features
            features = []
            for col in self.feature_columns:
                if col in df_copy.columns:
                    features.append(df_copy[col].values)
                else:
                    # Use zeros for missing columns
                    features.append(np.zeros(len(df_copy)))
            
            # Stack features and scale
            features = np.column_stack(features)
            scaled_features = self.feature_scaler.transform(features)
            
            return scaled_features
            
        except Exception as e:
            logging.error(f"Error preparing data point: {str(e)}")
            # Return a default array with the right shape
            return np.zeros((len(df), len(self.feature_columns)))
