import streamlit as st
import pandas as pd

from ui.prep import load_model

model = load_model('../models/model.pkl')

query = st.text_input("Your query", value="I love Streamlit!")
if query:
    pred = model.predict(pd.Series(['Testing my model i love it']))[0]
    result = 'Positive' if pred == 1 else 'Negative'
    st.write(result)