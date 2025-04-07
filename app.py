
import streamlit as st
import joblib

st.set_page_config(page_title='Spam Classifier', page_icon='📨')
st.title("📨 Spam Email Classifier")
st.write("Classify emails as spam or not spam using a Naive Bayes model.")

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input
email_text = st.text_area("Enter email text to classify:")

# Predict
if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        email_vector = vectorizer.transform([email_text])
        pred_proba = model.predict_proba(email_vector)[0][1]
        threshold = 0.4
        label = "🚫 Spam" if pred_proba > threshold else "✅ Not Spam"

        st.subheader("Prediction Result:")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Spam Probability:** {pred_proba:.2f}")
