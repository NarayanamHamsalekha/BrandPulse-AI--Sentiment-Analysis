# app.py
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import joblib
import os

# Load models
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
logistic_model = joblib.load("logistic_model.pkl")
naive_bayes_model = joblib.load("naive_bayes.pkl")

# Initialize Dash
app = dash.Dash(__name__)
server = app.server  # expose the Flask server

# Sample sentiment data
df = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Count": [10, 5, 3]
})

fig = px.pie(df, names="Sentiment", values="Count", title="Sentiment Distribution")

# Layout
app.layout = html.Div([
    html.H1("BrandPulse AI - Twitter Sentiment"),
    dcc.Graph(figure=fig),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
