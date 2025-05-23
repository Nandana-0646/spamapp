
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detection")

new_email = st.text_area("Enter the email text:")

if st.button("Classify Email"):
    if new_email:
        new_email_counts = vectorizer.transform([new_email])
        pred_prob = model.predict_proba(new_email_counts)
        if pred_prob.shape[0] > 0 and pred_prob.shape[1] > 1:
            result = "Spam" if pred_prob[0][1] > 0.4 else "Not Spam"
        else:
            result = "Unable to classify. Please check the input email."
        st.write(f"**Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter an email to classify.")
