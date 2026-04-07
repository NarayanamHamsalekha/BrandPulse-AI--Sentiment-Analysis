import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load the saved TF-IDF vectorizer and model
# -----------------------------
st.title("BrandPulse AI - Twitter Sentiment Analysis (Classical NLP)")

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load trained classifier (Logistic Regression or Naive Bayes)
clf_model = joblib.load("sentiment_model.pkl")

# -----------------------------
# Function to predict sentiment
# -----------------------------
def predict_sentiment(text):
    # Transform text to TF-IDF features
    features = tfidf_vectorizer.transform([text])
    # Predict using the classical model
    pred = clf_model.predict(features)
    return pred[0]  # returns 'Positive', 'Negative', or 'Neutral'

# -----------------------------
# Streamlit interface
# -----------------------------
st.write("Enter a tweet to predict its sentiment:")

user_input = st.text_area("Tweet text", height=100)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")

# -----------------------------
# Optional: simulate sentiment feed (example)
# -----------------------------
st.write("---")
st.subheader("Example Tweets")
sample_tweets = [
    "The service was amazing!",
    "I waited 4 hours just to get a cold burger.",
    "Not sure how I feel about this product.",
]

for tweet in sample_tweets:
    st.write(f"Tweet: {tweet} --> Sentiment: {predict_sentiment(tweet)}")
