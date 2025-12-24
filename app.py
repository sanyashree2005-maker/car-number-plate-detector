import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Car Number Plate Detection", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- OCR FUNCTION ----------------
def read_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(processed, config=config)

    return text.strip()

# ---------------- UI ----------------
st.title("🚗 Car Number Plate Detection using YOLO + OCR")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # ---------------- DETECTION ----------------
    results = model(img_np)[0]

    if results.boxes is not None:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            st.subheader("Detected Plate")
            st.image(plate, width=400)

            text = read_plate_text(plate)

            if text:
                st.success(f"Detected Text: {text}")
            else:
                st.warning("OCR could not read the plate clearly.")
    else:
        st.warning("No license plate detected.")
