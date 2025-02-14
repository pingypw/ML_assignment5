import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_model.pkl")

# Streamlit UI
st.title("My first ML App (Study on Imbalanced Data Classification sick_euthyroid dataset by 67130701703 ")

# Input fields
features = []

title = st.text_input("Text Input, Enter 42 features with comma separated")
st.write("example; feature_1,feature_2, .... , feature_n")
if len(title.split(',')) > 30:
    features = [float(i) for i in title.split(',')]

# Prediction
if st.button("Predict") and len(features) == 42:
    prediction = model.predict([np.array(features)])
    st.write(f"Predicted Class: {prediction[0]}")