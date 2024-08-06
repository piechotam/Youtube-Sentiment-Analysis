import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score

def load_features(files: list) -> tuple:
    """
    Load features from a list of files using pickle.

    Args:
    files (List): A list of file paths to load the features from.

    Returns:
    List: A list of features loaded from the given files.

    Raises:
    FileNotFoundError: If any of the files do not exist.
    IOError: If there is an error reading any of the files.
    pickle.UnpicklingError: If there is an error unpickling the file content.
    """
    features = []
    
    if not files:
        raise ValueError("The input list 'files' is empty.")
    
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        try:
            with open(file, 'rb') as f:
                feats = pickle.load(f)
                features.append(feats)
        except (IOError, pickle.UnpicklingError) as e:
            raise e
    
    return features

def test_models(models_to_test: list, X_train, y_train, X_valid, y_valid) -> dict: 
    performance = {}
    
    plt.figure(figsize=(8, 8))
    plt.title("ROC Curve for Different Models")
    
    for model in models_to_test:
        model.fit(X_train, y_train)
        model_name = model.__class__.__name__

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_valid)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_valid)
        else:
            print(f"{model_name} does not have `predict_proba` or `decision_function` method.")
            continue

        fpr, tpr, _ = roc_curve(y_valid, y_score)
        auc = round(roc_auc_score(y_valid, y_score), 4)
        plt.plot(fpr, tpr, label=f"{model_name}, AUC={auc}")

        y_pred = model.predict(X_valid)
        performance[model_name] = {
            'Accuracy': accuracy_score(y_valid, y_pred),
            'F1_score': f1_score(y_valid, y_pred)
        }

    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return performance