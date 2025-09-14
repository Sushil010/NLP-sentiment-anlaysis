import streamlit as st
import joblib
import pandas as pd
emo = {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
pipeline=joblib.load("sentiment_model.pkl")

st.title("Sentiment Prediction")

text=st.text_area("Enter text to predict: ")

if st.button("Predict"):
    X=pd.DataFrame([text],columns=["text"])
    prediction=pipeline.predict(X)[0]
    value=[k for k,v in emo.items() if v==prediction]
    st.write("Predicted Emotion: ",value[0])