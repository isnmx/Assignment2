import pandas as pd
import numpy as np

class DataProcessor:
    """Data processing class for loading and cleaning the bank customer data"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        self.n_samples = 0
        
    def load_data(self):
        """Load data from CSV file"""
        self.data = pd.read_csv(self.filepath)
        return self.data
    
    def clean_data(self):
        """Clean and prepare the data with sparse response handling"""
        # Convert date columns
        date_cols = [col for col in self.data.columns if 'date' in col.lower()]
        for col in date_cols:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        
        # Handle special values
        self.data.replace('.', np.nan, inplace=True)
        
        # Convert targets to numeric
        self.data['GOOD'] = pd.to_numeric(self.data['GOOD'], errors='coerce')
        self.data['BAD'] = pd.to_numeric(self.data['BAD'], errors='coerce')
        
        # Create ternary target with explicit missing handling
        self.data['target'] = np.select(
            [
                self.data['BAD'] == 1,
                self.data['GOOD'] == 1,
                self.data[['GOOD', 'BAD']].isna().all(axis=1)
            ],
            [
                0,  # BAD
                1,  # GOOD
                np.nan  # Explicit missing (will be filtered for training)
            ],
            default=np.nan  # Handle other cases as missing
        )
        
        # Filter for training (maintain original behavior)
        self.data = self.data.dropna(subset=['target'])
        
        # Remove original target columns
        self.data.drop(['GOOD', 'BAD'], axis=1, inplace=True)
        
        self.n_samples = len(self.data)
        if self.n_samples == 0:
            raise ValueError("No samples with valid target values found after cleaning.")
            
        return self.data

    def prepare_features_target(self):
        """Prepare features and target variables"""
        # Remove ID column and target
        self.X = self.data.drop(['target'], axis=1)
        self.y = self.data['target'].astype(int)
        
        return self.X, self.y
    
    def get_numeric_categorical_cols(self):
        """Identify numeric and categorical columns"""
        numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        
        # Special handling for occ_code which should be categorical
        if 'occ_code' in numeric_cols:
            numeric_cols = numeric_cols.drop('occ_code')
            categorical_cols = categorical_cols.union(['occ_code'])
        
        return numeric_cols, categorical_cols