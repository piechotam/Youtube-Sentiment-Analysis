import streamlit as st
import numpy as np

from ui.prep import load_model
from api.comments_extraction import extract_video_comments

model = load_model('models/model.pkl')

key = st.text_input('Provide your API key')
id = st.text_input('Provide video id')

if key and id:
    comments = extract_video_comments(API_KEY=key, VIDEO_ID=id, order='relevance')
    predictions = model.predict(comments)
    n_samples = predictions.size

    number_of_positive = np.sum(predictions)
    number_of_negative = n_samples - number_of_positive
    
    st.write(f'Out of {n_samples} comments the model detected {number_of_positive} positive comments and {number_of_negative} negative.')

    if st.button('Show negative comments'):
        indices_negative = np.where(predictions == 0)[0]
        st.write(comments[indices_negative])
    if st.button('Show positive comments'):
        indices_positive = np.where(predictions == 1)[0]
        st.write(comments[indices_positive])