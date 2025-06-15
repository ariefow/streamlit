# import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

# 
import streamlit as st
from deepface import DeepFace
import cv2
import requests                                                                 
import numpy as np                                                                                                                                   
from PIL import Image

# Daftar Model, Detector, dan Metric yang akan dicoba
models = [
    "VGG-Face", "Facenet", "Facenet512", "ArcFace",
]

backends = [
    "opencv", "yolov8", "yunet", "centerface",
]

#, "dlib"

metrics = ["cosine", "euclidean", "euclidean_l2"]


st.title("Aplikasi Verifikasi Wajah Menggunakan DeepFace")

st.write("Silakan unggah 2 foto yang ingin Anda bandingkan:")

# File upload
img1_file = st.file_uploader("Unggah foto pertama", type=["jpg","jpeg","png"])
img2_file = st.file_uploader("Unggah foto kedua", type=["jpg","jpeg","png"])

if st.button("Verifikasi Wajah") and img1_file and img2_file:
    st.success("Memulai proses verifikasi...")
    img1_path = "temp_img1.jpg"
    img2_path = "temp_img2.jpg"

    with open(img1_path, "wb") as f:
        f.write(img1_file.read())    

    with open(img2_path, "wb") as f:
        f.write(img2_file.read())    

    model_terbaik = None
    detector_terbaik = None
    metric_terbaik = None
    distance_terbaik = float('inf')  # Distance lebih dekat lebih bagus
    threshold_terbaik = float('inf')
    verified_terbaik = False

    for model in models:
        for detector in backends:
            for metric in metrics:
                st.write(f"Memverifikasi -> Model: {model}, Detector: {detector}, Metric: {metric}")

                try:
                    result = DeepFace.verify(
                        img1_path,
                        img2_path,
                        model_name=model,
                        detector_backend=detector,
                        align = True,
                        distance_metric=metric,
                        enforce_detection=False
                    )

                    st.write(f"✅ Hasil: Verified = {result['verified']}, Distance = {result['distance']:.4f}, Threshold = {result['threshold']}")

                    if result['verified'] and result['distance'] < distance_terbaik:
                        distance_terbaik = result['distance']
                        model_terbaik = model
                        detector_terbaik = detector
                        metric_terbaik = metric
                        threshold_terbaik = result['threshold']
                        verified_terbaik = result['verified']

                except Exception as e:
                    st.error(f"Gagal: {e}")

    st.success("Verifikasi Selesai")

    st.write("Hasil Terbaik:")
    st.write(f"Model   : {model_terbaik}")
    st.write(f"Detector: {detector_terbaik}")
    st.write(f"Metric  : {metric_terbaik}")
    st.write(f"Distance: {distance_terbaik}")
    st.write(f"Threshold: {threshold_terbaik}")
    st.write(f"Verified: {verified_terbaik}")

    st.info("Model dan konfigurasi disimpan di 'model_deepface.pt'")
    with open("model_deepface.pt", "w") as f:
        f.write(f"model_name = {model_terbaik}\n")
        f.write(f"detector = {detector_terbaik}\n")
        f.write(f"metric = {metric_terbaik}\n")
        f.write(f"distance = {distance_terbaik}\n")
        f.write(f"threshold = {threshold_terbaik}\n")
        f.write(f"verified = {verified_terbaik}\n")


