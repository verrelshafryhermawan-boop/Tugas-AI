# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="KNN - Teen Mental Health", layout="wide")

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
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    return model, accuracy, y_test, y_pred

def main():
    st.title("🧠 Aplikasi Klasifikasi Kesehatan Mental Remaja")
    st.write("Menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi tingkat depresi remaja.")

    df = load_data()
    model, accuracy, y_test, y_pred = train_model(df)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🎯 Prediksi", "📊 Diagram & Evaluasi", "📋 Dataset"])

    # ── TAB 1: PREDIKSI ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Masukkan Data untuk Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Umur", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
            social_media = st.slider("Jam Media Sosial per Hari", float(df.daily_social_media_hours.min()), float(df.daily_social_media_hours.max()), float(df.daily_social_media_hours.mean()))
            sleep = st.slider("Jam Tidur per Hari", float(df.sleep_hours.min()), float(df.sleep_hours.max()), float(df.sleep_hours.mean()))

        with col2:
            stress = st.slider("Tingkat Stres (1-10)", int(df.stress_level.min()), int(df.stress_level.max()), int(df.stress_level.mean()))
            anxiety = st.slider("Tingkat Kecemasan (1-10)", int(df.anxiety_level.min()), int(df.anxiety_level.max()), int(df.anxiety_level.mean()))

        if st.button("🔍 Prediksi Sekarang", use_container_width=True):
            input_data = np.array([[age, social_media, sleep, stress, anxiety]])
            prediction = model.predict(input_data)[0]
            st.success(f"🧬 Prediksi Tingkat Depresi: **{prediction}**")
            st.info(f"📈 Akurasi model: **{accuracy:.2f}%**")

    # ── TAB 2: DIAGRAM ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Evaluasi & Visualisasi Model")

        # Metrik Akurasi
        st.metric(label="Akurasi Model KNN", value=f"{accuracy:.2f}%")
        st.divider()

        col1, col2 = st.columns(2)

        # 1. Confusion Matrix
        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        # 2. Distribusi Label
        with col2:
            st.markdown("#### Distribusi Kelas Depression Label")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            df['depression_label'].value_counts().plot(kind='bar', ax=ax2, color=['#4C72B0','#DD8452','#55A868'])
            ax2.set_title("Distribusi Depression Label")
            ax2.set_xlabel("Label")
            ax2.set_ylabel("Jumlah")
            ax2.tick_params(axis='x', rotation=0)
            st.pyplot(fig2)

        st.divider()
        col3, col4 = st.columns(2)

        # 3. Korelasi Heatmap
        with col3:
            st.markdown("#### Heatmap Korelasi Fitur")
            features = ['age', 'daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level']
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
            ax3.set_title("Korelasi Antar Fitur")
            st.pyplot(fig3)

        # 4. Akurasi KNN vs K
        with col4:
            st.markdown("#### Akurasi KNN untuk Berbagai Nilai K")
            features = ['age', 'daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level']
            X = df[features]
            y = df['depression_label']
            X_train, X_test, y_train, y_test_k = train_test_split(X, y, test_size=0.2, random_state=42)
            k_values = list(range(1, 21))
            accuracies = []
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                accuracies.append(accuracy_score(y_test_k, knn.predict(X_test)) * 100)
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.plot(k_values, accuracies, marker='o', color='steelblue')
            ax4.axvline(x=5, color='red', linestyle='--', label='K=5 (dipakai)')
            ax4.set_title("Akurasi vs Nilai K")
            ax4.set_xlabel("Nilai K")
            ax4.set_ylabel("Akurasi (%)")
            ax4.legend()
            st.pyplot(fig4)

        st.divider()

        # 5. Distribusi fitur per label
        st.markdown("#### Distribusi Fitur per Depression Label")
        feature_option = st.selectbox("Pilih Fitur:", 
            ['age', 'daily_social_media_hours', 'sleep_hours', 'stress_level', 'anxiety_level'])
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        for label in df['depression_label'].unique():
            subset = df[df['depression_label'] == label][feature_option]
            ax5.hist(subset, alpha=0.6, label=label, bins=20)
        ax5.set_title(f"Distribusi {feature_option} per Label")
        ax5.set_xlabel(feature_option)
        ax5.set_ylabel("Frekuensi")
        ax5.legend()
        st.pyplot(fig5)

    # ── TAB 3: DATASET ────────────────────────────────────────────────────────
    with tab3:
        st.subheader("📋 Dataset Teen Mental Health")
        st.write(f"Total data: **{len(df)} baris**, **{len(df.columns)} kolom**")
        st.dataframe(df, use_container_width=True)
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
