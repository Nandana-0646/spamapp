
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detection")

new_email = st.text_area("Enter the email text:")

if st.button("Classify Email"):
    if new_email:
        new_email_counts = vectorizer.transform([new_email])
        pred_prob = model.predict_proba(new_email_counts)
        # Check if pred_prob has the expected shape before accessing elements
        if pred_prob.shape[0] > 0 and pred_prob.shape[1] > 1:  
            result = "Spam" if pred_prob[0][1] > 0.4 else "Not Spam"
        else:
            result = "Unable to classify. Please check the input email."  # Handle unexpected shape
        st.write(f"**Prediction:** {result}")
    else:
        st.write("⚠️ Please enter an email to classify.")
