import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from data_process import DataProcessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer, DefaultPredictor
import config

def main():
    try:
        # 1. Data loading and processing
        print("Loading and processing data...")
        processor = DataProcessor(config.FILEPATH)
        data = processor.load_data()
        data = processor.clean_data()
        
        # Save cleaned data
        print("\nSaving cleaned data to clean_data.csv...")
        data.to_csv('clean_data.csv', index=False)
        
        print(f"\nNumber of samples with valid target values: {processor.n_samples}")
        
        if processor.n_samples < 100:
            print("Warning: Very few samples with valid target values. Model may not perform well.")
        
        X, y = processor.prepare_features_target()
        numeric_cols, categorical_cols = processor.get_numeric_categorical_cols()
        
        # 2. Feature engineering
        print("\nPerforming feature engineering...")
        feature_engineer = FeatureEngineer(numeric_cols, categorical_cols)
        preprocessor = feature_engineer.create_preprocessor()
        
        # 3. Split data into train and test sets
        if processor.n_samples >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE, 
                stratify=y
            )
        else:
            print("\nNot enough samples for train-test split. Using all data for training.")
            X_train, y_train = X, y
            X_test, y_test = pd.DataFrame(), pd.Series()
        
        # 4. Model training
        print("\nTraining model...")
        trainer = ModelTrainer(preprocessor)
        model = trainer.train_model(X_train, y_train, random_state=config.RANDOM_STATE)
        
        # 5. Model evaluation
        if not X_test.empty:
            print("\nEvaluating model on test set...")
            test_metrics = trainer.evaluate_model(X_test, y_test)
        else:
            print("\nNo test set available for evaluation.")
            test_metrics = None
        
        # 6. Save model
        trainer.save_model(config.MODEL_SAVE_PATH)
        
        # 7. Example predictions
        print("\nMaking sample predictions...")
        predictor = DefaultPredictor(config.MODEL_SAVE_PATH)
        
        sample_data = X_train.head(5) if not X_train.empty else X.head(5)
        if not sample_data.empty:
            predictions = predictor.predict(sample_data)
            print("\nSample predictions:")
            print(predictions)
        return model, test_metrics
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    original_stdout = sys.stdout
    try:
        with open('result.log', 'w') as f:
            sys.stdout = f
            model, metrics = main()
    finally:    
        sys.stdout = original_stdout