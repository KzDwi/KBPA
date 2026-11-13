"""
Aplikasi Streamlit untuk Klasifikasi Spesies Ikan
Menggunakan Custom CNN yang telah dilatih dari awal (from scratch).

Jalankan dengan:
    streamlit run app.py
"""

import io
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import keras
from keras.layers import Dense

# ------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Spesies Ikan",
    page_icon="🐟",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fish_cnn_model.keras"
LABEL_MAP_PATH = BASE_DIR / "class_indices.json"
PREPROCESS_PATH = BASE_DIR / "preprocess_config.json"

_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init


# ------------------------------------------------------------
# Load model & konfigurasi (cache agar tidak reload setiap interaksi)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_configs():
    with open(LABEL_MAP_PATH, "r") as f:
        class_indices = json.load(f)  # {class_name: index}
    with open(PREPROCESS_PATH, "r") as f:
        preprocess_config = json.load(f)
    # Urutkan nama kelas berdasarkan indeks
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names, preprocess_config


model = load_model()
class_names, preprocess_config = load_configs()

IMG_SIZE = tuple(preprocess_config["image_size"])  # (224, 224)
RESCALE = preprocess_config["rescale_factor"]      # 1/255


# ------------------------------------------------------------
# Fungsi fetch gambar dari URL
# ------------------------------------------------------------
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
REQUEST_TIMEOUT = 10  # detik
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _looks_like_image_url(url: str) -> bool:
    """Cek cepat berdasarkan ekstensi (bukan jaminan, hanya filter awal)."""
    path = urlparse(url).path.lower()
    return path.endswith(ALLOWED_EXT)


def fetch_image_from_url(url: str) -> Image.Image:
    """
    Mengunduh gambar dari URL dan mengembalikan objek PIL.Image.
    Melempar ValueError jika URL bukan gambar valid.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL tidak boleh kosong.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL harus diawali dengan http:// atau https://")

    headers = {
        # Beberapa server menolak request tanpa User-Agent
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True
        )
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Gagal mengunduh gambar: {e}")

    if response.status_code != 200:
        raise ValueError(
            f"Server mengembalikan status {response.status_code}. "
            "Pastikan link dapat diakses publik."
        )

    # Validasi Content-Type
    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(
            f"URL bukan gambar (Content-Type: {content_type or 'tidak diketahui'}). "
            "Gunakan link langsung ke file gambar."
        )

    # Batasi ukuran agar tidak memori-overflow
    content = b""
    for chunk in response.iter_content(chunk_size=8192):
        content += chunk
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Ukuran gambar melebihi batas {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )

    try:
        image = Image.open(io.BytesIO(content))
        image.load()  # paksa decode untuk memastikan file benar-benar gambar
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"File tidak dapat dibaca sebagai gambar: {e}")

    return image.convert("RGB")


# ------------------------------------------------------------
# Fungsi prediksi
# ------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessing identik dengan saat training."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) * RESCALE
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr


def predict(image: Image.Image):
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]  # shape (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


# ------------------------------------------------------------
# Fungsi UI bersama (menampilkan hasil prediksi)
# ------------------------------------------------------------
def show_prediction_result(image: Image.Image, caption: str):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=caption, use_column_width=True)

    with st.spinner("Memprediksi..."):
        pred_label, confidence, probs = predict(image)

    with col2:
        st.subheader("Hasil Prediksi")
        st.success(f"**{pred_label}**")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.divider()
    st.subheader("Probabilitas per Kelas")
    st.caption(
        "Semakin panjang bar, semakin tinggi keyakinan model untuk kelas tersebut."
    )

    sorted_pairs = sorted(
        zip(class_names, probs),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, prob in sorted_pairs:
        st.progress(float(prob), text=f"{name} — {prob * 100:.2f}%")

    st.divider()
    st.caption(
        "⚠️ Model dilatih untuk mengenali kelas yang ada pada dataset. "
        "Gambar di luar domain dapat menghasilkan prediksi dengan confidence tinggi yang keliru."
    )


# ------------------------------------------------------------
# UI Utama
# ------------------------------------------------------------
st.title("🐟 Klasifikasi Spesies Ikan")
st.markdown(
    "Unggah gambar ikan **atau tempelkan URL gambar dari internet**, "
    "dan model **Custom CNN** (dilatih dari awal) akan memprediksi "
    "spesiesnya beserta tingkat keyakinan per kelas."
)

tab_upload, tab_url = st.tabs(["📁 Unggah File", "🔗 Dari Link / URL"])

# --- Tab 1: Upload file ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Pilih gambar ikan (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader_file",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            show_prediction_result(image, "Gambar yang diunggah")
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
    else:
        st.info("Silakan unggah gambar untuk memulai prediksi.")


# --- Tab 2: Dari URL ---
with tab_url:
    st.markdown(
        "Tempelkan **link langsung ke file gambar** "
        "(contoh: `https://example.com/ikan.jpg`). "
        "Link halaman web biasa (misal halaman Google Images) tidak akan bekerja."
    )

    url_input = st.text_input(
        "URL gambar",
        placeholder="https://contoh.com/gambar-ikan.jpg",
        key="url_input",
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        fetch_clicked = st.button("🔍 Ambil & Prediksi", type="primary")

    # Simpan hasil fetch di session_state agar tidak re-fetch saat interaksi lain
    if fetch_clicked and url_input.strip():
        try:
            with st.spinner("Mengunduh gambar dari URL..."):
                fetched_image = fetch_image_from_url(url_input)
            st.session_state["fetched_image"] = fetched_image
            st.session_state["fetched_url"] = url_input.strip()
        except ValueError as e:
            st.session_state.pop("fetched_image", None)
            st.session_state.pop("fetched_url", None)
            st.error(f"❌ {e}")
        except Exception as e:
            st.session_state.pop("fetched_image", None)
            st.session_state.pop("fetched_url", None)
            st.error(f"❌ Terjadi kesalahan tak terduga: {e}")

    # Tampilkan hasil jika ada gambar yang berhasil di-fetch
    if "fetched_image" in st.session_state:
        st.caption(f"Sumber: {st.session_state.get('fetched_url', '-')}")
        show_prediction_result(
            st.session_state["fetched_image"],
            "Gambar dari URL",
        )
    elif not fetch_clicked:
        st.info("Masukkan URL gambar lalu klik **Ambil & Prediksi**.")