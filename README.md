# 📊 Analisis Sentimen Kebijakan Kenaikan Gaji DPR

Aplikasi berbasis web untuk mengklasifikasikan sentimen publik terhadap kebijakan kenaikan gaji DPR menggunakan **Naive Bayes**.

🔗 **Live Demo**: [https://uts-eyeconic-sm8vbejuehewfb56s8j7vv.streamlit.app/](https://uts-eyeconic-sm8vbejueh6j7vv.streamlit.app/)

---

## 🎯 Tujuan
- Mengklasifikasikan komentar publik menjadi **Negatif**, **Netral**, atau **Positif**
- Menyediakan antarmuka pengguna yang mudah digunakan untuk analisis sentimen real-time
- Mengatasi tantangan umum seperti deteksi kalimat netral yang sering diabaikan sistem lain

---

## 🧠 Teknologi & Library
- **Bahasa Pemrograman**: Python
- **Framework ML**: Scikit-learn (`MultinomialNB`, `TfidfVectorizer`)
- **Preprocessing Teks**: Sastrawi (stemming), NLTK (stopwords), regex
- **Antarmuka Web**: Streamlit
- **Deployment**: Streamlit Community Cloud

---

## 🚀 Cara Menjalankan Lokal (Opsional)
Jika ingin menjalankan di komputer sendiri:

```bash
# 1. Clone repo ini
git clone https://github.com/Hendrik-Chandra-Kasih/Uts-Eyeconic.git
cd Uts-Eyeconic

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py

## Anggota Tim
221110026 - Hendrik Chandra Kasih
221110248 - Mhd Fadlurrahman Suhaemy
221113072 - Ribca Juliana Panjaitan

