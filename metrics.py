import numpy as np


def metrics(y, y_pred):
    """
    Determines overall model performance using the following five metrics:
    model accuracy, sensitivity, specificty, precision, and F1-score.

    Parameters:
        y (pd.Series): Actual target values
        y_pred (pd.Series): Predicted target values

    Returns:
        tuple: Model performance metrics (accuracy, sensitivity, specificity
    """

    # Calculate confusion matrix values
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y != 1) & (y_pred != 1))
    fp = np.sum((y != 1) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred != 1))

    # Calculate model accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate sensitivity
    sensitivity = tp / (tp + fn)

    # Calculate specificity
    # Note: Conditional avoids any divide by zero warnings
    if (tn + fp) == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)

    # Calculate precision
    precision = tp / (tp + fp)

    # Calculate f1-score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, sensitivity, specificity, precision, f1
