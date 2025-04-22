import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Model training and evaluation class"""
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.best_params = None
        
    def train_model(self, X, y, random_state=5117):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=random_state,
                class_weight='balanced'
            ))])
        
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return self.model
    
    def evaluate_model(self, X, y):
        """Evaluate model performance"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("\nROC AUC Score:", roc_auc_score(y, y_proba))
        print("Accuracy:", accuracy_score(y, y_pred))
        
        return classification_report(y, y_pred, output_dict=True)
    
    def save_model(self, filepath):
        """Save trained model to file"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

class DefaultPredictor:
    """Class for making predictions with the trained model"""
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        
    def predict(self, new_data, threshold=0.5, margin=0.1):
        """
        Make predictions on new data with uncertainty handling
        
        Parameters:
        - new_data: DataFrame containing the features
        - threshold: Decision threshold (default 0.5)
        - margin: Confidence margin for uncertain predictions (default 0.1)
        
        Returns:
        DataFrame with predictions, probabilities, and confidence categories
        """
        proba = self.model.predict_proba(new_data)[:, 1]

        lower_bound = threshold - margin
        upper_bound = threshold + margin

        decisions = []
        confidence_levels = []
        for p in proba:
            if p >= upper_bound:
                decisions.append('Accept')
                confidence_levels.append('High')
            elif p <= lower_bound:
                decisions.append('Decline')
                confidence_levels.append('High')
            else:
                decisions.append('Pass')
                confidence_levels.append('Low')
        results = pd.DataFrame({
            'Probability': proba,
            'Prediction': (proba >= threshold).astype(int),
            'Decision': decisions,
            'Confidence': confidence_levels,
            'Adjusted_Prediction': np.select(
                [proba >= upper_bound, proba <= lower_bound],
                [1, 0],
                default=-1
            )
        })
        
        return results