# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('Tuning-Model-NB.pkl', 'rb'))
    return vectorizer, model

# ===========================
# Preprocessing Functions
# ===========================
def remove_text_special(text):
    text = re.sub(r'\$\w*', '', str(text))
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'http\S+', "", text)
    text = re.sub(r'#', '', text)
    text = text.strip(' "\'')
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("\n", ' ').replace("Diterjemahkan oleh Google", ' ').replace("Asli", ' ')
    text = text.replace("http://", " ").replace("https://", " ")
    return text.strip()

def remove_emoji(text):
    # Versi aman: hapus semua non-ASCII
    return text.encode('ascii', 'ignore').decode('ascii')

def stopword_removal(tokens):
    filtering = stopwords.words('indonesian')
    return [token for token in tokens if token not in filtering]

def text_preprocessing(text):
    text = remove_text_special(text)
    text = remove_emoji(text)
    tokens = word_tokenize(text.lower())
    tokens = stopword_removal(tokens)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed).strip()

# ===========================
# Streamlit App
# ===========================
def main():
    st.set_page_config(page_title="UTS ML EYECONIC Sentimen Kebijakan DPR", page_icon="🏛️")
    st.title("🔍 Analisis Sentimen: Kenaikan Gaji DPR")
    st.markdown("Masukkan komentar untuk analisis sentimen.")

    try:
        vectorizer, model = load_model()
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan vectorizer.pkl dan Tuning-Model-NB.pkl ada.")
        st.stop()

    user_input = st.text_area("💬 Komentar Anda:", height=150, placeholder='Contoh: "Saya netral aja..."')

    if st.button("🧠 Analisis Sentimen"):
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu!")
        else:
            prep_text = text_preprocessing(user_input)
            text_vec = vectorizer.transform([prep_text])
            pred_label = model.predict(text_vec)[0]
            proba = model.predict_proba(text_vec)[0]
            classes = model.classes_
            idx = list(classes).index(pred_label)
            confidence = proba[idx] * 100

            color_map = {'negatif': '#dc3545', 'netral': '#6c757d', 'positif': '#198754'}
            color = color_map.get(pred_label, '#000000')

            st.subheader("Hasil:")
            st.markdown(
                f"""
                <div style="text-align:center; padding:20px; border-radius:10px; background-color:{color}20; border:2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{pred_label.capitalize()}</h2>
                    <p style="font-size:1.3em; margin:10px 0;">{confidence:.1f}% keyakinan</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Lihat teks setelah preprocessing"):
                st.code(prep_text)

if __name__ == "__main__":
    main()