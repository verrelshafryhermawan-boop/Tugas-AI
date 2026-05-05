# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN - Teen Mental Health", layout="centered")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/verrelshafryhermawan-boop/Tugas-AI/refs/heads/main/Teen_Mental_Health_Dataset.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def train_model(df):
    features = ['age', 'daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level']
    X = df[features]
    y = df['depression_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, accuracy

def main():
    st.title("🧠 Aplikasi Klasifikasi Kesehatan Mental Remaja")
    st.write("Masukkan nilai fitur berikut untuk memprediksi tingkat depresi:")

    df = load_data()
    model, accuracy = train_model(df)

    age = st.slider("Umur", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
    social_media = st.slider("Jam Media Sosial per Hari", float(df.daily_social_media_hours.min()), float(df.daily_social_media_hours.max()), float(df.daily_social_media_hours.mean()))
    sleep = st.slider("Jam Tidur per Hari", float(df.sleep_hours.min()), float(df.sleep_hours.max()), float(df.sleep_hours.mean()))
    stress = st.slider("Tingkat Stres (1-10)", int(df.stress_level.min()), int(df.stress_level.max()), int(df.stress_level.mean()))
    anxiety = st.slider("Tingkat Kecemasan (1-10)", int(df.anxiety_level.min()), int(df.anxiety_level.max()), int(df.anxiety_level.mean()))

    if st.button("Prediksi"):
        input_data = np.array([[age, social_media, sleep, stress, anxiety]])
        prediction = model.predict(input_data)[0]
        st.success(f"🧬 Prediksi: **{prediction}**")
        st.info(f"Akurasi model pada data uji: {accuracy:.2f}%")

    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
