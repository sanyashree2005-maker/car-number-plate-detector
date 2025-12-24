import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image

st.set_page_config(page_title="Car Number Plate Detector", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚗 Car Number Plate Detection & OCR")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

def read_number_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = model(img_np)[0]

    if results.boxes:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            text = read_number_plate(plate)

            st.image(plate, caption="Detected Plate", width=300)
            st.success(f"Detected Text: **{text}**")
    else:
        st.warning("No number plate detected.")
