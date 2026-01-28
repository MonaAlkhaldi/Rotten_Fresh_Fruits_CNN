# app.py
import io
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import CNN

# =========================
# CONFIG
# =========================
IMG_SIZE = 224

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

# نفس preprocessing حق الاختبار (بدون augmentation)
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = CNN()
    state = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# =========================
# PREDICT
# =========================
def predict(image, model):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
    return probs

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🍎 Fresh vs Rotten Fruits", layout="centered")

st.title("🍎 Fresh vs Rotten Fruit Detection")
st.caption("CNN-based image classification demo")

model = load_model()

uploaded = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    probs = predict(image, model)
    pred = int(np.argmax(probs))

    st.subheader("Prediction")
    st.write(f"**{CLASS_NAMES[pred]}**")
    st.write(f"Confidence: **{probs[pred]*100:.2f}%**")

    st.subheader("Class Probabilities")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p*100:.2f}%")

