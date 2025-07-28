import streamlit as st
import pandas as pd
import joblib
import requests
import io

st.set_page_config(page_title="Deteksi Credit Card Fraud", layout="wide")
st.title("💳 Deteksi Credit Card Fraud")
st.write("Unggah file CSV yang berisi data transaksi untuk mendeteksi apakah ada kecurangan (fraud).")

@st.cache_resource
def load_model():
    url = "https://huggingface.co/stvn1809/fraud-detector/resolve/main/random_forest_best1.pkl"
    response = requests.get(url)
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()

def preprocess(df):
    df_clean = df.copy()  # salin data untuk tetap menjaga data asli

    # Proses kolom waktu
    if 'TransactionDate' in df_clean.columns:
        df_clean['TransactionDate'] = pd.to_datetime(df_clean['TransactionDate'], errors='coerce')
        df_clean['year'] = df_clean['TransactionDate'].dt.year
        df_clean['month'] = df_clean['TransactionDate'].dt.month
        df_clean['day'] = df_clean['TransactionDate'].dt.day
        df_clean['hour'] = df_clean['TransactionDate'].dt.hour
        df_clean['minute'] = df_clean['TransactionDate'].dt.minute
        df_clean['second'] = df_clean['TransactionDate'].dt.second
        df_clean.drop(columns=['TransactionDate'], inplace=True)

    # Drop TransactionID
    if 'TransactionID' in df_clean.columns:
        df_clean.drop(columns=['TransactionID'], inplace=True)

    # Simpan label jika ada
    y = None
    if 'IsFraud' in df_clean.columns:
        y = df_clean['IsFraud']
        df_clean.drop(columns=['IsFraud'], inplace=True)

    # Encode kategorikal
    if 'TransactionType' in df_clean.columns:
        df_clean['TransactionType'] = df_clean['TransactionType'].astype('category').cat.codes
    if 'Location' in df_clean.columns:
        df_clean['Location'] = df_clean['Location'].astype('category').cat.codes

    return df_clean, y

uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Awal")
        st.dataframe(df_raw)

        # Simpan salinan untuk ditampilkan nanti
        df_display = df_raw.copy()

        # Preprocessing
        X, y_true = preprocess(df_raw)

        # Pastikan urutan kolom sesuai model
        model_features = list(model.feature_names_in_)
        missing_cols = [col for col in model_features if col not in X.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut hilang atau tidak cocok: {missing_cols}")
            st.stop()
        X = X[model_features]

        # Prediksi
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Tambahkan hasil prediksi ke data awal
        df_display['Label Prediksi'] = pd.Series(y_pred).map({0: 'Normal', 1: 'Fraud'})
        df_display['Probabilitas Fraud (%)'] = (y_proba[:, 1] * 100).round(2)

        # Tampilkan hasil
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df_display)

        # Ringkasan jumlah
        fraud_total = (y_pred == 1).sum()
        normal_total = (y_pred == 0).sum()
        st.success(f"✅ Transaksi Normal: {normal_total}")
        st.error(f"🚨 Transaksi Fraud: {fraud_total}")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca atau memproses file: {e}")
