import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image

st.set_page_config(page_title="Car Number Plate Detection", layout="wide")

st.title("🚗 Car Number Plate Detection using YOLO + OCR")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # MUST exist in repo root

model = load_model()

def preprocess_for_ocr(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(img_np, conf=0.4)[0]

    if results.boxes is None or len(results.boxes) == 0:
        st.error("No number plate detected")
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            st.image(plate, caption="Detected Plate", width=300)

            plate_bin = preprocess_for_ocr(plate)

            text = pytesseract.image_to_string(
                plate_bin,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            text = text.strip().replace("\n", "")
            st.success(f"Detected Text: {text if text else 'Not readable'}")
