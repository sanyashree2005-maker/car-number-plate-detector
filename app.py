import streamlit as st
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Car Number Plate Detection", layout="wide")
st.title("🚗 Car Number Plate Detection using YOLO + OCR")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- OCR FUNCTION ----------------
def read_number_plate(plate):
    h, w = plate.shape[:2]

    # Crop central text area
    plate = plate[int(0.25*h):int(0.75*h), int(0.05*w):int(0.95*w)]

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    custom_config = r"""
    --oem 3
    --psm 7
    -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
    """

    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip().replace(" ", ""), thresh

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # YOLO detection
    results = model(img_np, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        st.error("No number plate detected.")
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            text, processed_plate = read_number_plate(plate)

            st.subheader("Detected Plate")
            st.image(plate, width=350)

            st.subheader("Processed Plate for OCR")
            st.image(processed_plate, width=350)

            st.success(f"Detected Text: {text if text else 'Unable to read'}")

        
