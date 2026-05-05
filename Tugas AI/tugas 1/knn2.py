# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("KNN - Teen Mental Health Dataset")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Silakan upload file Teen_Mental_Health_Dataset.csv")
    st.stop()

st.success("File ditemukan! Memproses data...")

# Membaca data
df = pd.read_csv(uploaded_file)

# Persiapan Data
features = ['age', 'daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level']
X = df[features]
y = df['depression_label']

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Hasil
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred) * 100

st.write("### Hasil Model KNN")
st.metric(label="Akurasi Model", value=f"{akurasi:.2f}%")
