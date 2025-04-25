import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
from data_process import DataProcessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import config

def main():
    try:
        # 1. Data loading and processing
        print("Loading and processing data...")
        processor = DataProcessor(config.FILEPATH)
        data = processor.load_data()
        data = processor.clean_data()
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

        # 3. Model setup and cross-validation
        print("\nInitializing model and cross-validation...")
        trainer = ModelTrainer(preprocessor)
        base_model = trainer.create_model(random_state=config.RANDOM_STATE)
        kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        
        accuracies = []
        f1_scores = []
        
        # 4. Cross-validation loop
        print("\nStarting cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = clone(base_model)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            accuracies.append(acc)
            f1_scores.append(f1)
            print(f"Fold {fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # 5. Cross-validation visualization
        print("\nGenerating validation metrics visualization...")
        plt.figure(figsize=(10, 5))
        x = np.arange(len(accuracies))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score')
        plt.xticks(x, [f'Fold {i+1}' for i in range(len(accuracies))])
        plt.ylabel('Scores')
        plt.title('Cross-Validation Performance Metrics')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig('cross_val_metrics.png')
        plt.close()

        # 6. Final model training
        print("\nTraining final model on full dataset...")
        final_model = clone(base_model)
        final_model.fit(X, y)
        trainer.save_model(config.MODEL_SAVE_PATH)

        # 7. Generate sample predictions
        sample_preds = final_model.predict(X.head(100))
        print("\nSample predictions from final model:")
        print(sample_preds)

        # Return metrics
        metrics = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
        
        return final_model, metrics

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