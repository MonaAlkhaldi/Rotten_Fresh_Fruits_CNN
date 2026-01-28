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
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="🍎 Fruit Freshness AI",
    page_icon="🍏",
    layout="centered"
)

# =========================
# CONFIG
# =========================
CLASS_NAMES = [
    "fresh apples 🍎",
    "fresh bananas 🍌",
    "fresh oranges 🍊",
    "rotten apples 🤢",
    "rotten bananas 🤮",
    "rotten oranges 🤧"
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
    <h1 style="text-align:center;">🍎 Fruit Freshness AI</h1>
    <p style="text-align:center; color:gray;">
    Upload a fruit image and let the AI judge its freshness 🍏🤖
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
    st.image(image, caption="📸 Your fruit", use_container_width=True)

    with st.spinner("🧠 Thinking like a fruit expert..."):
        probs = predict(image, model)
        pred = int(np.argmax(probs))
        confidence = probs[pred] * 100

    st.divider()

    # =========================
    # TEXT-ONLY PREDICTION (BUT FUN 😄)
    # =========================
    if confidence > 85:
        tone = "The model is very confident"
    elif confidence > 65:
        tone = "The model is fairly confident"
    else:
        tone = "The model is unsure, but thinks"

    st.markdown(
        f"""
        <h3 style="text-align:center;">
        🧾 Prediction Result
        </h3>

        <p style="font-size:18px; text-align:center;">
        {tone} that this image shows <b>{CLASS_NAMES[pred]}</b>.
        </p>

        <p style="text-align:center; color:gray;">
        Confidence level: {confidence:.2f}%
        </p>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("⬆️ Upload a fruit image to see the AI prediction.")

st.divider()

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <p style="text-align:center; color:gray; font-size:14px;">
    🍏 Built with PyTorch & Streamlit • Educational project
    </p>
    """,
    unsafe_allow_html=True
)

