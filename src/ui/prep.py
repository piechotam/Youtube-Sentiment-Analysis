import streamlit as st
import pickle

@st.cache_resource
def load_model(file: str):
    """
    Load a model from a file.

    Parameters:
    file (str): The path to the file containing the model.

    Returns:
    The loaded model.
    """
    try:
        with open(file, 'rb') as f:
            model = pickle.load(f)
        return model
    except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
        print(f"An error occurred while loading the model: {e}")
        return None