import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_descriptors(y_true, y_pred):
    # Ensure y_pred is in the correct format (class labels)
    if y_pred.ndim > 1:  # if y_pred comes as probabilities
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:  # converting from one-hot encoded if necessary
        y_true = np.argmax(y_true, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

