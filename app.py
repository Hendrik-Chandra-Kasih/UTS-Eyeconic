# app.py
import streamlit as st
import pickle
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Download stopwords sekali (aman di cloud)
nltk.download('stopwords', quiet=True)

# ===========================
# Load Model & Vectorizer
# ===========================
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('Tuning-Model-NB.pkl', 'rb'))
    return vectorizer, model

# ===========================
# Preprocessing Functions (Tanpa word_tokenize!)
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
    return text.encode('ascii', 'ignore').decode('ascii')

def text_preprocessing(text):
    text = remove_text_special(text)
    text = remove_emoji(text)
    # ✅ GUNAKAN .split() BUKAN word_tokenize
    tokens = text.lower().split()
    # Stopword removal (aman di cloud)
    id_stopwords = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in id_stopwords]
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed).strip()

# ===========================
# Aplikasi Streamlit
# ===========================
def main():
    st.set_page_config(page_title="Sentimen Kebijakan DPR", page_icon="🏛️")
    st.title("🔍 Analisis Sentimen: Kenaikan Gaji DPR")
    st.markdown("Masukkan komentar publik untuk analisis sentimen.")

    # Load model
    try:
        vectorizer, model = load_model()
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan vectorizer.pkl dan Tuning-Model-NB.pkl ada.")
        st.stop()

    # Input teks
    user_input = st.text_area("💬 Komentar Anda:", height=150, placeholder='Contoh: "Saya netral aja..."')

    if st.button("🧠 Analisis Sentimen"):
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu!")
        else:
            # Preprocessing & prediksi
            prep_text = text_preprocessing(user_input)
            text_vec = vectorizer.transform([prep_text])
            pred_label = model.predict(text_vec)[0]
            proba = model.predict_proba(text_vec)[0]
            classes = model.classes_
            idx = list(classes).index(pred_label)
            confidence = proba[idx] * 100

            # Warna berdasarkan label
            color_map = {'negatif': '#dc3545', 'netral': '#6c757d', 'positif': '#198754'}
            color = color_map.get(pred_label, '#000000')

            # Tampilkan hasil
            st.subheader("Hasil Analisis:")
            st.markdown(
                f"""
                <div style="text-align:center; padding:20px; border-radius:10px; 
                            background-color:{color}20; border:2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{pred_label.capitalize()}</h2>
                    <p style="font-size:1.3em; margin:10px 0;">{confidence:.1f}% keyakinan</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Opsional: lihat preprocessing
            with st.expander("Lihat teks setelah preprocessing"):
                st.code(prep_text)

# ===========================
# Jalankan Aplikasi
# ===========================
if __name__ == "__main__":
    main()
