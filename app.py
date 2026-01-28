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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🍎 Fruit Freshness Detector",
    page_icon="🍏",
    layout="centered"
)

# =========================
# CONFIG
# =========================
CLASS_NAMES = [
    "Fresh Apples 🍎",
    "Fresh Bananas 🍌",
    "Fresh Oranges 🍊",
    "Rotten Apples 🤢",
    "Rotten Bananas 🤮",
    "Rotten Oranges 🤧"
]

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
        probs = F.softmax(model(x), dim=1)[0].numpy()
    return probs

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center;'>🍎 Fruit Freshness Detector</h1>
    <p style='text-align: center; color: gray;'>
    AI-powered CNN model to detect fresh vs rotten fruits
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# APP BODY
# =========================
model = load_model()

uploaded = st.file_uploader(
    "📤 Upload a fruit image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    st.image(
        image,
        caption="📸 Uploaded Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing image..."):
        probs = predict(image, model)
        pred = int(np.argmax(probs))
        confidence = probs[pred]

    st.success("✅ Prediction Complete")

    # =========================
    # RESULT CARD
    # =========================
    st.markdown(
        f"""
        <div style="
            background-color:#f9f9f9;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        ">
            <h2>{CLASS_NAMES[pred]}</h2>
            <p style="font-size:18px;">
                Confidence: <b>{confidence*100:.2f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # CONFIDENCE BAR
    # =========================
    st.markdown("### 🔵 Confidence Level")
    st.progress(float(confidence))

else:
    st.info("⬆️ Please upload a fruit image to get started.")

st.divider()

# =========================
# FOOTER
# =========================
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built with ❤️ using PyTorch & Streamlit"
    "</p>",
    unsafe_allow_html=True
)


