import streamlit as st
import pandas as pd
import joblib
import requests
import io
from sklearn.metrics import classification_report

# === Judul Aplikasi ===
st.set_page_config(page_title="Deteksi Credit Card Fraud", layout="wide")
st.title("💳 Deteksi Credit Card Fraud")
st.write("Unggah file CSV yang berisi data transaksi untuk mendeteksi apakah ada kecurangan (fraud).")

# === Load model dari Hugging Face ===
@st.cache_resource
def load_model():
    url = "https://huggingface.co/stvn1809/fraud-detector/resolve/main/random_forest_best1.pkl"
    response = requests.get(url)
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()

def preprocess(df):
    if 'TransactionDate' in df.columns:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        df['year'] = df['TransactionDate'].dt.year
        df['month'] = df['TransactionDate'].dt.month
        df['day'] = df['TransactionDate'].dt.day
        df['hour'] = df['TransactionDate'].dt.hour
        df['minute'] = df['TransactionDate'].dt.minute
        df['second'] = df['TransactionDate'].dt.second
        df.drop(columns=['TransactionDate'], inplace=True)

    if 'TransactionID' in df.columns:
        df.drop(columns=['TransactionID'], inplace=True)

    y = None
    if 'IsFraud' in df.columns:
        y = df['IsFraud']
        df.drop(columns=['IsFraud'], inplace=True)

    if 'TransactionType' in df.columns:
        df['TransactionType'] = df['TransactionType'].astype('category').cat.codes
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype('category').cat.codes

    return df, y

# === Upload File ===
uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Awal")
        st.dataframe(df_raw.head())

        # Preprocessing
        X, y_true = preprocess(df_raw)

        # Cek kecocokan kolom
        model_features = list(model.feature_names_in_)
        missing_cols = [col for col in model_features if col not in X.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut hilang atau tidak cocok: {missing_cols}")
            st.stop()

        X = X[model_features]

        # Prediksi
        y_pred = model.predict(X)
        df_raw['Prediksi'] = y_pred
        df_raw['Label Prediksi'] = df_raw['Prediksi'].map({0: 'Normal', 1: 'Fraud'})

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df_raw[['Prediksi', 'Label Prediksi']].head())

        # Ringkasan
        fraud_total = (df_raw['Prediksi'] == 1).sum()
        normal_total = (df_raw['Prediksi'] == 0).sum()
        st.success(f"✅ Transaksi Normal: {normal_total}")
        st.error(f"🚨 Transaksi Fraud: {fraud_total}")

        if y_true is not None:
            st.subheader("📋 Evaluasi Model (Jika Label Ada)")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.json(report)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca atau memproses file: {e}")
