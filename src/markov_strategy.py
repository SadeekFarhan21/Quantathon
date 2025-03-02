import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import defaultdict

class MarkovStrategy:
    """
    Implements a Markov chain strategy for market predictions
    based on transition probabilities between market states.
    """
    
    def __init__(self, data):
        """
        Initialize the Markov strategy
        
        Args:
            data (DataFrame): Market data with states
        """
        self.data = data.copy()
        self.transition_matrix = None
        self.state_encoder = None
        self.predictions = None
        
    def train_model(self, state_column='Market_State', training_window=252):
        """
        Train the Markov chain model using market state transitions
        
        Args:
            state_column (str): Column containing market states
            training_window (int): Number of days to use for training
            
        Returns:
            dict: Transition probabilities
        """
        if state_column not in self.data.columns:
            logging.error(f"State column '{state_column}' not found in data")
            if 'Predicted_Market' in self.data.columns:
                state_column = 'Predicted_Market'
                logging.info(f"Using 'Predicted_Market' column instead")
            else:
                logging.error("No suitable state column found for Markov model")
                return None
        
        # Use the provided training_window
        df = self.data.iloc[-training_window:].copy() if training_window else self.data.copy()
        
        # Encode market states to integers
        self.state_encoder = LabelEncoder()
        df['Encoded_State'] = self.state_encoder.fit_transform(df[state_column])
        
        # Calculate state transitions
        transitions = defaultdict(lambda: defaultdict(int))
        states = df['Encoded_State'].unique()
        state_names = self.state_encoder.classes_
        
        # Count transitions
        for i in range(len(df) - 1):
            current_state = df['Encoded_State'].iloc[i]
            next_state = df['Encoded_State'].iloc[i + 1]
            transitions[current_state][next_state] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for current in transitions:
            total = sum(transitions[current].values())
            transition_probs[current] = {next_state: count / total for next_state, count in transitions[current].items()}
        
        # Store transition matrix for predictions
        self.transition_matrix = transitions
        self.transition_probs = transition_probs
        self.states = states
        self.state_names = state_names
        
        # Log transition probabilities
        for current_idx, current in enumerate(state_names):
            for next_idx, next_state in enumerate(state_names):
                prob = transition_probs.get(current_idx, {}).get(next_idx, 0)
                logging.debug(f"P({current} → {next_state}) = {prob:.2f}")
        
        return transition_probs
        
    def generate_predictions(self):
        """
        Generate predictions using trained Markov model
        
        Returns:
            DataFrame: Data with predictions added
        """
        if self.transition_matrix is None or self.state_encoder is None:
            logging.error("Markov model not trained. Call train_model() first.")
            return self.data
            
        # Create predictions dataframe
        df = self.data.copy()
        if 'Market_State' in df.columns:
            state_column = 'Market_State'
        elif 'Predicted_Market' in df.columns:
            state_column = 'Predicted_Market'
        else:
            logging.error("No suitable state column found in data")
            return df
        
        # Encode market states
        df['Encoded_State'] = self.state_encoder.transform(df[state_column])
        
        # Initialize prediction columns
        df['Next_State_Prediction'] = None
        df['Prediction_Confidence'] = 0.0
        
        # Generate predictions for each day
        for i in range(len(df) - 1):
            current_state = df['Encoded_State'].iloc[i]
            
            # Get transition probabilities for current state
            next_probs = self.transition_probs.get(current_state, {})
            if not next_probs:
                continue
            
            # Predict next state based on transition probabilities
            next_state = max(next_probs, key=next_probs.get)
            confidence = next_probs[next_state]
            
            # Decode predicted state
            df.at[i, 'Next_State_Prediction'] = self.state_encoder.inverse_transform([next_state])[0]
            df.at[i, 'Prediction_Confidence'] = confidence
        
        self.predictions = df
        return df
