import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
from ultralytics import YOLO

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Car Number Plate Detection",
    layout="wide"
)

st.title("🚗 Car Number Plate Detection using YOLO + OCR")

# -------------------- LOAD YOLO MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------- OCR CLEANING --------------------
def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

# -------------------- IMAGE PREPROCESSING FOR OCR --------------------
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # -------------------- YOLO DETECTION --------------------
    results = model(image_np, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        st.error("❌ No number plate detected.")
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            plate = image_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            st.subheader("Detected Plate")
            st.image(plate, use_column_width=False)

            # -------------------- OCR --------------------
            plate_bin = preprocess_plate(plate)

            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            raw_text = pytesseract.image_to_string(plate_bin, config=config)

            final_text = clean_plate_text(raw_text)

            st.success(f"Detected Text: {final_text if final_text else 'Unreadable'}")
