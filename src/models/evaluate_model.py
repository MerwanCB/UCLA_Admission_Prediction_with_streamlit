from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification predictions with accuracy and confusion matrix.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Calculating evaluation metrics...")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm