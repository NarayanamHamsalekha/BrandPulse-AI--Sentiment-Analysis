# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# -----------------------
# Paths
# -----------------------
MODEL_DIR = "models"
DATA_FILE = "cleaned_tweets.csv"

# -----------------------
# Load Models & Tokenizers
# -----------------------
st.sidebar.title("Model Loading Status")

def safe_load(file_path, loader_func=joblib.load):
    try:
        obj = loader_func(file_path)
        st.sidebar.success(f"✅ Loaded: {os.path.basename(file_path)}")
        return obj
    except:
        st.sidebar.warning(f"⚠️ Not Found: {os.path.basename(file_path)}")
        return None

lr_model = safe_load(f"{MODEL_DIR}/logistic_model.pkl")
nb_model = safe_load(f"{MODEL_DIR}/naive_bayes_model.pkl")
tfidf = safe_load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
lstm_model = safe_load(f"{MODEL_DIR}/lstm_model.keras", loader_func=load_model)
tokenizer = safe_load(f"{MODEL_DIR}/tokenizer.pkl")
le = safe_load(f"{MODEL_DIR}/label_encoder.pkl")

# -----------------------
# Load Dataset
# -----------------------
try:
    df = pd.read_csv(DATA_FILE)
    st.sidebar.success(f"✅ Dataset Loaded")
except:
    df = pd.DataFrame(columns=['text', 'sentiment'])
    st.sidebar.warning(f"⚠️ Dataset Not Found")

# -----------------------
# Streamlit Layout
# -----------------------
st.title("🚀 BrandPulse AI - Twitter Sentiment Dashboard")
st.markdown("### Analyze Twitter Sentiments in Real-Time")
st.write("---")

# -----------------------
# Text Cleaning Function
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Custom Tweet Input
# -----------------------
st.header("💬 Custom Tweet Sentiment Prediction")
user_input = st.text_area("Enter a Tweet:")

if st.button("Predict Sentiment"):
    if user_input:
        cleaned_input = clean_text(user_input)
        
        # Classical Predictions
        vec_input = tfidf.transform([cleaned_input]) if tfidf else None
        lr_pred = lr_model.predict(vec_input)[0] if lr_model else "N/A"
        nb_pred = nb_model.predict(vec_input)[0] if nb_model else "N/A"

        # LSTM Prediction
        if lstm_model and tokenizer and le:
            seq = tokenizer.texts_to_sequences([cleaned_input])
            pad_seq = pad_sequences(seq, maxlen=100)
            lstm_pred_class = np.argmax(lstm_model.predict(pad_seq), axis=1)
            lstm_pred_label = le.inverse_transform(lstm_pred_class)[0]
        else:
            lstm_pred_label = "N/A"

        st.success(f"✅ Final Prediction (LSTM): {lstm_pred_label}")
        st.markdown(f"**Naive Bayes Prediction:** {nb_pred}")
        st.markdown(f"**Logistic Regression Prediction:** {lr_pred}")
    else:
        st.warning("Please enter a tweet!")

st.write("---")

# -----------------------
# Live Tweet Stream (Simulation)
# -----------------------
st.header("📢 Live Tweet Stream (Simulation)")
if not df.empty:
    sample_tweets = df.sample(5)['text']
    for i, tweet in enumerate(sample_tweets, start=1):
        st.write(f"{i}. {tweet}")
else:
    st.info("No tweets to display.")

st.write("---")

# -----------------------
# Sentiment Distribution
# -----------------------
st.header("📊 Sentiment Distribution (LSTM Predictions)")
if lstm_model and tokenizer and le and not df.empty:
    df['text'] = df['text'].fillna("")
    X_seq = tokenizer.texts_to_sequences(df['text'])
    X_pad = pad_sequences(X_seq, maxlen=100)
    y_pred_all = np.argmax(lstm_model.predict(X_pad), axis=1)
    counts = pd.Series(y_pred_all).value_counts().sort_index()
    labels = le.inverse_transform([0,1,2])
    dist_df = pd.DataFrame({"Sentiment": labels, "Count": counts.values})
    st.bar_chart(dist_df.set_index('Sentiment'))
else:
    st.info("Sentiment distribution cannot be displayed.")

st.write("---")

# -----------------------
# Trend Line / Time-Series (Simulation)
# -----------------------
st.header("📈 Sentiment Trend Over Last 24 Hours (Simulation)")
timestamps = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
sentiment_counts = np.random.randint(0, 50, size=(24, 3))
trend_df = pd.DataFrame(sentiment_counts, columns=labels if le else ["neg","neu","pos"], index=timestamps)
st.line_chart(trend_df)

st.write("---")

# -----------------------
# Performance Metrics
# -----------------------
st.header("📑 Performance Comparison Table")
if le and not df.empty and lr_model and nb_model:
    y_true = le.transform(df['sentiment'])
    y_pred_lr_all = le.transform(lr_model.predict(tfidf.transform(df['text'])))
    y_pred_nb_all = le.transform(nb_model.predict(tfidf.transform(df['text'])))
    y_pred_lstm_all = np.argmax(lstm_model.predict(X_pad), axis=1) if lstm_model else np.zeros(len(df))
    
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Naive Bayes', 'LSTM'],
        'Accuracy': [
            accuracy_score(y_true, y_pred_lr_all),
            accuracy_score(y_true, y_pred_nb_all),
            accuracy_score(y_true, y_pred_lstm_all)
        ],
        'Precision': [
            precision_score(y_true, y_pred_lr_all, average='weighted'),
            precision_score(y_true, y_pred_nb_all, average='weighted'),
            precision_score(y_true, y_pred_lstm_all, average='weighted')
        ],
        'Recall': [
            recall_score(y_true, y_pred_lr_all, average='weighted'),
            recall_score(y_true, y_pred_nb_all, average='weighted'),
            recall_score(y_true, y_pred_lstm_all, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_true, y_pred_lr_all, average='weighted'),
            f1_score(y_true, y_pred_nb_all, average='weighted'),
            f1_score(y_true, y_pred_lstm_all, average='weighted')
        ]
    })
    st.dataframe(comparison)
else:
    st.info("Performance metrics cannot be displayed.")

st.write("---")

# -----------------------
# Confusion Matrices
# -----------------------
st.markdown("## Confusion Matrices (String Labels)")
if le and not df.empty:
    y_true_labels = le.inverse_transform(y_true)
    y_pred_lr_labels = le.inverse_transform(y_pred_lr_all)
    y_pred_nb_labels = le.inverse_transform(y_pred_nb_all)
    y_pred_lstm_labels = le.inverse_transform(y_pred_lstm_all)
    labels = ['negative', 'neutral', 'positive']

    def plot_confusion_matrix_altair(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index().melt(id_vars='index')
        cm_df.columns = ['Actual', 'Predicted', 'Count']
        
        chart = alt.Chart(cm_df).mark_rect().encode(
            x='Predicted:N',
            y='Actual:N',
            color='Count:Q',
            tooltip=['Actual', 'Predicted', 'Count']
        ).properties(
            width=400,
            height=300,
            title=f"{model_name} Confusion Matrix"
        )
        st.altair_chart(chart)

    plot_confusion_matrix_altair(y_true_labels, y_pred_lr_labels, "Logistic Regression")
    plot_confusion_matrix_altair(y_true_labels, y_pred_nb_labels, "Naive Bayes")
    plot_confusion_matrix_altair(y_true_labels, y_pred_lstm_labels, "LSTM")
