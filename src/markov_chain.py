import numpy as np
import pandas as pd
import logging

class MarkovChain:
    """
    Implementation of a Markov Chain model for market state transitions
    """
    def __init__(self):
        """Initialize the Markov Chain model"""
        self.states = None
        self.transition_matrix = None
        self.state_counts = None
        self.steady_state = None
        
    def fit(self, state_sequence):
        """
        Fit the Markov model to a sequence of states
        
        Args:
            state_sequence (array-like): Sequence of state labels
        """
        # Handle edge cases with empty or too short sequences
        if len(state_sequence) == 0:
            raise ValueError("Cannot fit Markov model on empty state sequence")
        
        if len(state_sequence) < 2:
            logging.warning("State sequence too short for transition probabilities")
            self.states = np.unique(state_sequence)
            n_states = len(self.states)
            # Create uniform transition matrix as fallback
            self.transition_matrix = pd.DataFrame(
                1/n_states, 
                index=self.states,
                columns=self.states
            )
            return self
        
        # Get unique states
        self.states = np.unique(state_sequence)
        n_states = len(self.states)
        
        # Create transition count matrix
        transitions = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(state_sequence) - 1):
            from_idx = np.where(self.states == state_sequence[i])[0][0]
            to_idx = np.where(self.states == state_sequence[i+1])[0][0]
            transitions[from_idx, to_idx] += 1
            
        # Create state counts
        self.state_counts = np.zeros(n_states)
        for i, state in enumerate(self.states):
            self.state_counts[i] = np.sum(state_sequence == state)
            
        # Calculate transition probabilities
        self.transition_matrix = pd.DataFrame(transitions, index=self.states, columns=self.states)
        
        # Normalize rows to get probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        
        # Handle zero rows (states with no outgoing transitions)
        for i, sum_val in enumerate(row_sums):
            if sum_val == 0:
                # If a state has no outgoing transitions, assume equal probability to all states
                self.transition_matrix.iloc[i] = 1 / n_states
            else:
                # Normalize to probabilities
                self.transition_matrix.iloc[i] = self.transition_matrix.iloc[i] / sum_val
                
        # Calculate steady state distribution
        self._calculate_steady_state()
        
        return self
    
    def _calculate_steady_state(self):
        """Calculate the steady state distribution using power iteration"""
        try:
            # Initialize uniform distribution
            n_states = len(self.states)
            v = np.ones(n_states) / n_states
            
            # Convert DataFrame to numpy array for faster calculation
            T = self.transition_matrix.values
            
            # Iterate until convergence or max iterations
            max_iter = 1000
            tolerance = 1e-6
            
            for _ in range(max_iter):
                v_new = v @ T
                
                # Check convergence
                if np.allclose(v, v_new, rtol=tolerance):
                    break
                    
                v = v_new
            
            # Create Series with state labels as index
            self.steady_state = pd.Series(v, index=self.states)
            
        except Exception as e:
            # Fallback to uniform distribution in case of error
            logging.warning(f"Error calculating steady state: {str(e)}. Using uniform distribution.")
            self.steady_state = pd.Series(1/len(self.states), index=self.states)
            
    def predict_next(self, current_state):
        """
        Predict the next state based on current state
        
        Args:
            current_state: Current state label
            
        Returns:
            Next state prediction
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if current_state not in self.transition_matrix.index:
            # Handle unknown state with fallback to most common state
            logging.warning(f"Unknown state: {current_state}. Using most probable state.")
            if self.steady_state is not None:
                return self.steady_state.idxmax()
            else:
                return self.states[0]  # Fallback to first state
        
        # Get transition probabilities for current state
        probs = self.transition_matrix.loc[current_state].values
        
        # Sample next state according to the probabilities
        next_state = np.random.choice(self.states, p=probs)
        
        return next_state
    
    def predict_sequence(self, initial_state, length=10):
        """
        Predict a sequence of states starting from an initial state
        
        Args:
            initial_state: Initial state label
            length (int): Length of sequence to predict
            
        Returns:
            list: Predicted sequence of states
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        sequence = [initial_state]
        current_state = initial_state
        
        for _ in range(length - 1):
            current_state = self.predict_next(current_state)
            sequence.append(current_state)
            
        return sequence
