from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, auc, precision_recall_curve
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
class ModelValidator:
    """Model validation class with time-series evaluation and calibration analysis"""
    def __init__(self, model, data):
        """
        Initialize validator with trained model and full dataset
        
        Args:
            model: Trained classifier
            data: DataFrame containing features and target (must include 'date' column)
        """
        self.model = model
        self.data = data
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']

    def time_series_validate(self):
        """Perform time-series cross-validation with metric tracking"""
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        
        for train_idx, test_idx in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics.append({
                'pr_auc': self._pr_auc(y_test, proba),
                'calibration_error': self._calibration_error(y_test, proba),
                'classification_report': classification_report(
                    y_test, 
                    (proba >= 0.5).astype(int), 
                    output_dict=True
                )
            })
            
        return pd.DataFrame(metrics)

    def _pr_auc(self, y_true, proba):
        """Calculate precision-recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, proba)
        return auc(recall, precision)

    def _calibration_error(self, y_true, proba, n_bins=10):
        """Calculate expected calibration error"""
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))

    def plot_calibration(self, y_true, proba):
        """Generate calibration plot with reliability diagram"""
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(prob_pred, prob_true, marker='o', label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Actual probability')
        ax.set_title('Model Calibration Curve')
        ax.legend()
        ax.grid(True)
        
        return fig
    def plot_roc_curve(self, y_true, proba):
            """Plot ROC curve with AUC score"""
            from sklearn.metrics import roc_curve,roc_auc_score
            import matplotlib.pyplot as plt
            
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_auc = roc_auc_score(y_true, proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            return plt.gcf()

    def plot_confusion_matrix(self, y_true, y_pred):
            """Plot annotated confusion matrix"""
            from sklearn.metrics import ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(7, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, 
                cmap=plt.cm.Blues, 
                ax=ax,
                display_labels=['Non-Default', 'Default']
            )
            ax.set_title('Confusion Matrix')
            return fig

    def plot_feature_importance(self, feature_names, top_n=20):
            """Plot feature importance from Random Forest model"""
            import matplotlib.pyplot as plt
            import numpy as np
            
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                # For pipeline models
                importances = self.model.named_steps['classifier'].feature_importances_
            
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            return plt.gcf()
