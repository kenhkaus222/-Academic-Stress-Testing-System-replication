"""
Data Cleaning Module for S&P 500 Time Series Analysis
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import List
from datetime import timedelta

class DataCleaner:
    """
    S&P 500 Data Cleaning Module
    Specialized for handling 20 years of ^GSPC data from multiple CSV files
    """
    
    def __init__(self, target_years: int = 20):
        self.target_years = target_years
        self.raw_data = None
        self.cleaned_data = None
        self.full_data = None  # Store full dataset before filtering
        
    def find_gspc_files(self, data_dir: str = ".") -> List[str]:
        """Find all GSPC CSV files"""
        patterns = [
            os.path.join(data_dir, "GSPC*.csv"),
            os.path.join(data_dir, "*GSPC*.csv"),
            os.path.join(data_dir, "^GSPC*.csv")
        ]
        
        files = []
        for pattern in patterns:
            found_files = glob.glob(pattern)
            if found_files:
                files.extend(found_files)
        
        files = sorted(list(set(files)))
        return files
    
    def load_and_merge(self, data_dir: str = ".") -> pd.DataFrame:
        """Load and merge all GSPC files"""
        files = self.find_gspc_files(data_dir)
        
        if not files:
            raise FileNotFoundError("No GSPC CSV files found")
        
        dfs = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                continue
        
        if not dfs:
            raise ValueError("Could not read any GSPC files")
        
        merged_df = pd.concat(dfs, ignore_index=True)
        self.raw_data = merged_df
        return merged_df
    
    def clean(self, df: pd.DataFrame = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Clean the S&P 500 data with optional date filtering
        
        Parameters:
        df: DataFrame to clean
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        """
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='first')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Clean price columns
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean volume
        if 'Volume' in df.columns:
            if df['Volume'].dtype == 'object':
                df['Volume'] = df['Volume'].str.replace(',', '')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Remove invalid prices
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Validate OHLC relationships
        if all(col in df.columns for col in price_columns):
            valid_ohlc = (
                (df['High'] >= df['Low']) & 
                (df['High'] >= df['Open']) & 
                (df['High'] >= df['Close']) & 
                (df['Low'] <= df['Open']) & 
                (df['Low'] <= df['Close'])
            )
            df = df[valid_ohlc]
        
        # Calculate log returns
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Remove extreme outliers (>20% daily move)
        outliers = (df['Return'] > 0.20) | (df['Return'] < -0.20)
        df = df[~outliers]
        
        # Store full data before date filtering
        self.full_data = df.copy()
        
        # Apply date filtering if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['Date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['Date'] <= end_dt]
        else:
            # Default: Filter for target years
            end_date = df['Date'].max()
            start_date = end_date - timedelta(days=int(self.target_years * 365.25))
            df = df[df['Date'] >= start_date]
        
        self.cleaned_data = df
        return df
    
    def get_returns(self) -> np.ndarray:
        """Get returns array for modeling"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available")
        return self.cleaned_data['Return'].dropna().values
    
    def find_crisis_dates(self, calibration_end_date: str = None, top_n: int = 3):
        """
        Find top N crisis dates (worst returns) after calibration period
        
        Parameters:
        calibration_end_date: End date of calibration period (YYYY-MM-DD)
        top_n: Number of crisis dates to find
        
        Returns:
        List of tuples (date, return) for top N crisis events
        """
        if self.full_data is None:
            raise ValueError("No data available. Run clean() first.")
        
        data = self.full_data.copy()
        
        # Filter for dates after calibration period if specified
        if calibration_end_date:
            calibration_end = pd.to_datetime(calibration_end_date)
            data = data[data['Date'] > calibration_end]
        
        # Sort by returns to find worst days
        data_sorted = data.nsmallest(top_n, 'Return')
        
        crisis_dates = []
        for idx, row in data_sorted.iterrows():
            crisis_dates.append({
                'date': row['Date'],
                'return': row['Return']
            })
        
        return crisis_dates