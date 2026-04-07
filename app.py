import streamlit as st
import joblib
import pandas as pd
import random
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="BrandPulse AI", layout="wide")
st.title("BrandPulse AI - Twitter Sentiment Analysis")

# -----------------------------
# Load TF-IDF vectorizer and models
# -----------------------------
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
logistic_model = joblib.load("logistic_model.pkl")
naive_bayes_model = joblib.load("naive_bayes.pkl")

# -----------------------------
# Predict sentiment function
# -----------------------------
def predict_sentiment(text, model_type="Logistic Regression"):
    features = tfidf_vectorizer.transform([text])
    if model_type == "Logistic Regression":
        pred = logistic_model.predict(features)
    else:
        pred = naive_bayes_model.predict(features)
    return pred[0]

# -----------------------------
# Simulated tweet feed
# -----------------------------
st.subheader("Live Tweet Stream")
sample_tweets = [
    "The service was amazing!",
    "I waited 4 hours for a cold burger.",
    "Not sure how I feel about this product.",
    "Absolutely loved the new phone!",
    "Terrible customer support, very disappointed.",
    "This is okay, nothing special.",
]

# User selects model
model_choice = st.radio("Select Model", ["Logistic Regression", "Naive Bayes"])

# Number of simulated tweets
num_tweets = st.slider("Number of incoming tweets to simulate", 5, 20, 10)

tweets_data = []
for _ in range(num_tweets):
    tweet_text = random.choice(sample_tweets)
    sentiment = predict_sentiment(tweet_text, model_choice)
    timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
    tweets_data.append({"tweet": tweet_text, "sentiment": sentiment, "time": timestamp})

df = pd.DataFrame(tweets_data).sort_values("time")

st.dataframe(df)

# -----------------------------
# Sentiment Distribution Pie Chart
# -----------------------------
st.subheader("Sentiment Distribution")
sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]
fig_pie = px.pie(sentiment_counts, values="Count", names="Sentiment", title="Sentiment Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Trend Over Time Line Chart
# -----------------------------
st.subheader("Trend Over Time (Last 1 Hour)")
trend_df = df.groupby([pd.Grouper(key="time", freq="5Min"), "sentiment"]).size().reset_index(name="count")
fig_line = px.line(trend_df, x="time", y="count", color="sentiment", title="Sentiment Trend Over Time")
st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# Custom tweet prediction
# -----------------------------
st.subheader("Predict Your Own Tweet")
user_input = st.text_area("Enter a tweet", height=100)
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter text to predict!")
    else:
        sentiment = predict_sentiment(user_input, model_choice)
        st.success(f"Predicted Sentiment ({model_choice}): {sentiment}")
