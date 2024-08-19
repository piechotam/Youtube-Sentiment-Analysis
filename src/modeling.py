import pickle
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline

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
    """
    Test multiple models and evaluate their performance using ROC curve, accuracy, and F1 score.

    Args:
        models_to_test (list): List of models to test.
        X_train: Training data.
        y_train: Training labels.
        X_valid: Validation data.
        y_valid: Validation labels.

    Returns:
        dict: A dictionary containing the performance metrics (accuracy and F1 score) for each tested model.
    """
    performance = {}

    plt.figure(figsize=(6, 6))
    plt.title("ROC Curve for Different Models")

    for model in models_to_test:
        start = time.time()
        model_name = model.__class__.__name__
        print(f'Fitting {model_name}...')
        model.fit(X_train, y_train)
        fitting_time = time.time() - start

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

        start = time.time()
        y_pred = model.predict(X_valid)
        pred_time = time.time() - start

        print(f'Fitting time: {fitting_time}\nPrediction time: {pred_time}')

        performance[model_name] = {
            'Accuracy': accuracy_score(y_valid, y_pred),
            'F1_score': f1_score(y_valid, y_pred)
        }

    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return performance

def print_performance(performance: dict) -> None:
    """
    Prints the performance metrics for each model.

    Args:
        performance (dict): A dictionary containing the performance metrics for each model.

    Returns:
        None
    """
    for model_name, metrics_dict in performance.items():
        print(f'================{model_name}================')
        for metric, value in metrics_dict.items():
            print(f'{metric}: {round(value, 4)}')
        
def hyperparameters_tuning(param_grid: dict, model, folds: int, param_comb: int, X, y, use_random_search: bool = True) -> None:
    """
    Perform hyperparameter tuning using either RandomizedSearchCV (default) or GridSearchCV.

    Args:
        param_grid (dict): Dictionary with hyperparameter names as keys and lists of values as values.
        model: The model to be tuned.
        folds (int): Number of cross-validation folds.
        param_comb (int): Number of parameter settings that are sampled.
        X: The input features.
        y: The target variable.
        use_random_search (bool, optional): If True, use RandomizedSearchCV. If False, use GridSearchCV. Defaults to True.

    Returns:
        search object
    """
    
    if use_random_search:
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=param_comb,
                                    scoring='roc_auc', random_state=42, cv=folds, n_jobs=-1)
    else:
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=folds, n_jobs=-1)
    
    search.fit(X, y)

    print("Best Hyperparameters:", search.best_params_)
    print("Best ROC_AUC:", search.best_score_)

    return search

def add_model_to_pipeline(pipeline_file: str, destination_file: str, model) -> Pipeline:
    """
    Adds a model to an existing pipeline and saves the updated pipeline to a destination file.

    Args:
        pipeline_file (str): The file path of the existing pipeline to load.
        destination_file (str): The file path to save the updated pipeline.
        model: The model to add to the pipeline.

    Returns:
        The updated processing pipeline.
    """
    with open(pipeline_file, 'rb') as f:
        processing_pipeline = pickle.load(f)

    if not isinstance(processing_pipeline, Pipeline):
        raise ValueError("The loaded object is not a valid sklearn Pipeline.")

    processing_pipeline.steps.append(('classifier', model))

    with open(destination_file, 'wb') as f:
        pickle.dump(processing_pipeline, f)
    
    return processing_pipeline
