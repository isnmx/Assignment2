from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer:
    """Feature engineering class for preprocessing the data"""
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.preprocessor = None
        
    def create_preprocessor(self):
            """Create preprocessing pipeline"""
            # Numeric features pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            
            # Categorical features pipeline
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            # Combine preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_cols),
                    ('cat', categorical_transformer, self.categorical_cols)])
            
            return self.preprocessor
