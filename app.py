import streamlit as st
import pandas as pd
import joblib
import requests
import io
from sklearn.metrics import classification_report
import shap

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
explainer = shap.TreeExplainer(model)

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

uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Awal")
        st.dataframe(df_raw.head())

        X, y_true = preprocess(df_raw)

        model_features = list(model.feature_names_in_)
        missing_cols = [col for col in model_features if col not in X.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut hilang atau tidak cocok: {missing_cols}")
            st.stop()

        X = X[model_features]

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        df_raw['Prediksi'] = y_pred
        df_raw['Label Prediksi'] = df_raw['Prediksi'].map({0: 'Normal', 1: 'Fraud'})
        df_raw['Probabilitas Fraud (%)'] = (y_proba[:, 1] * 100).round(2)

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df_raw[['Prediksi', 'Label Prediksi', 'Probabilitas Fraud (%)']].head())

        fraud_total = (df_raw['Prediksi'] == 1).sum()
        normal_total = (df_raw['Prediksi'] == 0).sum()
        st.success(f"✅ Transaksi Normal: {normal_total}")
        st.error(f"🚨 Transaksi Fraud: {fraud_total}")

        if y_true is not None:
            st.subheader("📋 Evaluasi Model (Jika Label Ada)")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.json(report)

        st.subheader("🔍 Alasan Prediksi untuk Transaksi Tertentu")
        selected_index = st.number_input("Pilih indeks baris transaksi (0 - {})".format(len(X)-1), min_value=0, max_value=len(X)-1, value=0)

        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame({
        'Fitur': list(X.columns),
        'Nilai Fitur': X.iloc[selected_index].values,
        'Kontribusi SHAP': shap_values[1][selected_index]
        }).sort_values(by='Kontribusi SHAP', key=abs, ascending=False)

        st.write(f"📌 Penjelasan prediksi untuk baris ke-{selected_index}:")
        st.dataframe(shap_df.head(5))

        st.info("Kontribusi SHAP menunjukkan seberapa besar pengaruh masing-masing fitur terhadap keputusan model. Nilai positif mendorong prediksi ke arah 'Fraud', sedangkan negatif ke arah 'Normal'.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca atau memproses file: {e}")
