import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract

st.set_page_config(page_title="Car Number Plate Detection", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚗 Car Number Plate Detection using YOLO + OCR")

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "png", "jpeg"]
)

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(img_np)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            processed = preprocess_plate(plate)

            text = pytesseract.image_to_string(
                processed,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            st.image(plate, caption="Detected Plate", width=300)
            st.success(f"Detected Text: {text.strip()}")
    else:
        st.warning("No number plate detected.")

