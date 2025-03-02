import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDataLoader:
    def __init__(self, data_path='data/market_data.xlsx'):
        """
        Load market data from the provided Excel file.
        
        Args:
            data_path (str): Path to the Excel file with market data
        """
        self.data_path = data_path
        self.price_data = None
        self.prob_data = None
        
    def load_data(self):
        """
        Load and process the market data.
        
        Returns:
            tuple: (price_df, prob_df) - DataFrames for price data and probability data
        """
        try:
            logging.info(f"Loading market data from {self.data_path}")
            
            # Read the Excel file directly using the exact column names
            # Skip rows that might contain descriptions
            df = pd.read_excel(self.data_path, header=None)
            
            # Find the row with the headers
            header_row = None
            for i in range(min(15, len(df))):  # Look at first 15 rows at most
                # Check if this row contains our expected headers
                row_values = df.iloc[i].astype(str).str.strip().tolist()
                row_text = " ".join(row_values).lower()
                if "date" in row_text and "s&p500" in row_text and "bond rate" in row_text:
                    header_row = i
                    break
            
            if header_row is None:
                logging.error("Could not find header row with 'Date', 'S&P500', 'Bond Rate'")
                # Try a fallback approach - assume first row
                header_row = 0
            
            # Read data with the detected header row
            logging.info(f"Reading data with header at row {header_row}")
            df = pd.read_excel(self.data_path, header=header_row)
            
            # Clean up column names to handle spaces and capitalization
            df.columns = [str(col).strip() for col in df.columns]
            
            # Print columns to debug
            logging.info(f"Found columns: {df.columns.tolist()}")
            
            # Check for the specific column names we expect
            price_cols = []
            prob_cols = []
            
            # The exact column names according to your description
            for col in df.columns:
                # Check for price data columns (first section)
                if col in ['Date', 'S&P500', 'Bond Rate']:
                    price_cols.append(col)
                # Check for probability columns (second section) - some might be duplicated
                elif col in ['Date.1', 'Date_1', 'PrDec', 'PrInc']:
                    prob_cols.append(col)
            
            # If we don't find exact matches, try case-insensitive matching
            if len(price_cols) < 2:  # We need at least Date and S&P500
                for col in df.columns:
                    col_lower = col.lower()
                    if 'date' in col_lower and col not in price_cols and col not in prob_cols:
                        price_cols.append(col)
                    elif 's&p' in col_lower or 'sp500' in col_lower:
                        price_cols.append(col)
                    elif 'bond' in col_lower:
                        price_cols.append(col)
                    elif 'prdec' in col_lower.replace(' ', '') or 'pr dec' in col_lower:
                        prob_cols.append(col)
                    elif 'princ' in col_lower.replace(' ', '') or 'pr inc' in col_lower:
                        prob_cols.append(col)
            
            logging.info(f"Identified price columns: {price_cols}")
            logging.info(f"Identified probability columns: {prob_cols}")
            
            # Handle case where we have more columns than we need
            if len(price_cols) > 3:
                price_cols = price_cols[:3]
            
            # Extract and process price data
            if len(price_cols) >= 3:  # Need at least Date, S&P500, Bond Rate
                price_df = df[price_cols].copy()
                # Rename columns to standardized names
                price_df.columns = ['Date', 'SP500', 'BondRate']
            else:
                # Handle case where we need to create the DataFrame from scratch
                logging.warning("Creating price data DataFrame from specific columns")
                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                sp_col = next((col for col in df.columns if 's&p' in col.lower() or 'sp500' in col.lower()), None)
                bond_col = next((col for col in df.columns if 'bond' in col.lower() or 'rate' in col.lower()), None)
                
                if not all([date_col, sp_col, bond_col]):
                    # One last attempt - try positional columns
                    price_df = df.iloc[:, :3].copy()
                    price_df.columns = ['Date', 'SP500', 'BondRate']
                else:
                    price_df = df[[date_col, sp_col, bond_col]].copy()
                    price_df.columns = ['Date', 'SP500', 'BondRate']
            
            # Extract and process probability data
            if len(prob_cols) >= 3:  # Need at least Date, PrDec, PrInc
                prob_df = df[prob_cols].copy()
                # Rename columns to standardized names
                prob_df.columns = ['Date', 'PrDec', 'PrInc']
            else:
                # Try to identify by position if columns E, F, G contain our probability data
                date2_col = next((col for col in df.columns if col not in price_cols and 'date' in col.lower()), None)
                prdec_col = next((col for col in df.columns if 'prdec' in col.lower().replace(' ', '') or 'dec' in col.lower()), None)
                princ_col = next((col for col in df.columns if 'princ' in col.lower().replace(' ', '') or 'inc' in col.lower()), None)
                
                if all([date2_col, prdec_col, princ_col]):
                    prob_df = df[[date2_col, prdec_col, princ_col]].copy()
                    prob_df.columns = ['Date', 'PrDec', 'PrInc']
                else:
                    # If we can't identify by name, try using columns 4, 5, 6 (0-indexed)
                    if len(df.columns) >= 7:  # Make sure we have enough columns
                        prob_df = df.iloc[:, 4:7].copy()
                        prob_df.columns = ['Date', 'PrDec', 'PrInc']
                    else:
                        raise ValueError("Could not identify probability data columns")
            
            # Convert dates and set index
            price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
            prob_df['Date'] = pd.to_datetime(prob_df['Date'], errors='coerce')
            
            # Drop rows with invalid dates
            price_df = price_df.dropna(subset=['Date'])
            prob_df = prob_df.dropna(subset=['Date'])
            
            # Set index
            price_df.set_index('Date', inplace=True)
            prob_df.set_index('Date', inplace=True)
            
            # Convert to numeric
            for col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
                
            for col in prob_df.columns:
                prob_df[col] = pd.to_numeric(prob_df[col], errors='coerce')
            
            # Drop rows with all NaN values
            price_df = price_df.dropna(how='all')
            prob_df = prob_df.dropna(how='all')
            
            # Store data
            self.price_data = price_df
            self.prob_data = prob_df
            
            logging.info(f"Loaded price data with {len(price_df)} rows and probability data with {len(prob_df)} rows")
            return price_df, prob_df
            
        except Exception as e:
            logging.error(f"Error loading market data: {str(e)}")
            raise
            
    def merge_data(self, fill_method='ffill'):
        """
        Merge price and probability data with improved date alignment
        """
        if self.price_data is None or self.prob_data is None:
            self.load_data()
        
        # Merge the two dataframes on Date index
        merged_df = pd.merge(self.price_data, self.prob_data, 
                             left_index=True, right_index=True, 
                             how='left')
        
        # Forward fill probability values (using ffill() instead of fillna(method=))
        if fill_method == 'ffill':
            merged_df[['PrDec', 'PrInc']] = merged_df[['PrDec', 'PrInc']].ffill()
        elif fill_method == 'bfill':
            merged_df[['PrDec', 'PrInc']] = merged_df[['PrDec', 'PrInc']].bfill()
        
        # Add extra validation to ensure dates are aligned correctly
        if merged_df['PrDec'].isna().sum() > 0:
            logging.warning(f"Found {merged_df['PrDec'].isna().sum()} missing values in PrDec after merge")
            
        # Try different interpolation methods if too many NaNs
        if merged_df['PrDec'].isna().mean() > 0.5:  # If more than 50% missing
            logging.warning("Too many missing values after merge, trying alternative interpolation")
            # Try to match closest dates instead
            for idx in merged_df.index[merged_df['PrDec'].isna()]:
                closest_date = self.prob_data.index[abs(self.prob_data.index - idx).argmin()]
                if abs((closest_date - idx).days) < 7:  # Within a week
                    merged_df.loc[idx, ['PrDec', 'PrInc']] = self.prob_data.loc[closest_date]
                    
        # Apply the requested fill method
        merged_df[['PrDec', 'PrInc']] = merged_df[['PrDec', 'PrInc']].fillna(method=fill_method)
        
        # Sort the index to ensure it's monotonic for slicing
        merged_df = merged_df.sort_index()
        
        return merged_df
        
    def get_data_for_period(self, start_date='2019-01-01', end_date='2022-12-31'):
        """
        Get market data for a specific time period.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            DataFrame: Market data for the specified period
        """
        merged_df = self.merge_data()
        
        # Ensure the index is sorted before slicing
        merged_df = merged_df.sort_index()
        
        # Convert dates if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Filter by date range - use .loc only if dates are in range
        # First check if the dates are within the index range
        if start_date and start_date > merged_df.index.max():
            logging.warning(f"Start date {start_date} is beyond available data range")
            return pd.DataFrame()
            
        if end_date and end_date < merged_df.index.min():
            logging.warning(f"End date {end_date} is before available data range")
            return pd.DataFrame()
        
        # Adjust dates to be within range if needed
        effective_start = max(start_date, merged_df.index.min()) if start_date else merged_df.index.min()
        effective_end = min(end_date, merged_df.index.max()) if end_date else merged_df.index.max()
        
        logging.info(f"Filtering data from {effective_start} to {effective_end}")
        
        try:
            # Use .loc with the adjusted date range
            filtered_df = merged_df.loc[effective_start:effective_end]
            return filtered_df
        except KeyError as e:
            # If slicing fails, try another approach
            logging.warning(f"Date slicing failed: {e}")
            filtered_df = merged_df[
                (merged_df.index >= effective_start) & 
                (merged_df.index <= effective_end)
            ]
            return filtered_df