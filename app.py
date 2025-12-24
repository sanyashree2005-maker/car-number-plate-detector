import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Number Plate Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚗 Car Number Plate Detection using YOLO")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = model(img_np)

    plate_crop = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img_np[y1:y2, x1:x2]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(img_np, caption="Detected Plate Region", use_column_width=True)

    if plate_crop is not None:
        st.subheader("Detected Plate")
        st.image(plate_crop, width=300)

        st.info(
            "OCR is disabled in Streamlit Cloud due to system limitations.\n\n"
            "Number plate text extraction is supported in local / Docker deployments."
        )
    else:
        st.warning("No number plate detected.")
