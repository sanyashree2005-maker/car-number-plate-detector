import streamlit as st
import numpy as np
import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Number Plate Detection", layout="centered")

st.title("🚗 Car Number Plate Detection using YOLO + OCR")

# -----------------------------
# Load YOLO model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# OCR function (robust)
# -----------------------------
def read_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

# -----------------------------
# Upload image
# -----------------------------
uploaded = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    results = model(img_np)[0]

    if len(results.boxes) == 0:
        st.warning("No number plate detected.")
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            st.subheader("Detected Plate")
            st.image(plate, use_column_width=False)

            text = read_plate_text(plate)

            if text:
                st.success(f"Detected Text: {text}")
            else:
                st.warning("Text not clearly readable")
