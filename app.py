import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_model.pkl")

# Streamlit UI
st.title("My first ML App (Study on Imbalanced Data Classification sick_euthyroid dataset by 67130701703 ")

# Input fields
features = []
title = '77,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1.2,0,1,0.6,0,1,71,0,1,0.68,0,1,104,1,0'
title = st.text_input("Text Input, Enter 42 features with comma separated")
st.write("example; feature_1,feature_2, .... , feature_n")
features = [float(i) for i in title.split(',')]

# Prediction
if st.button("Predict") and title != 'default':
    prediction = model.predict([np.array(features)])
    st.write(f"Predicted Class: {prediction[0]}")