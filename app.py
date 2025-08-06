import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🌐 Language Detection App")
st.write("Detects whether a sentence is in English,Malayalam,Hindi,Tamil, Kannada, French, Spanish, Portuguese, Italian, Sweedish, Dutch, Arabic, Turkish, German, Danish, Greek.")





user_input = st.text_input("Enter a sentence:")

if user_input:
    vectorized_text = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_text)
    st.success(f"Predicted Language: **{prediction[0]}**")