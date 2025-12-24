import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO
from PIL import Image

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# OCR function
# -------------------------------
def read_number_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)

    text = re.sub(r"[^A-Z0-9]", "", text)
    return text, thresh

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Car Number Plate Detection", layout="centered")

st.title("🚗 Car Number Plate Detection & OCR")
st.write("Upload an image to detect and read the vehicle number plate.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # YOLO Detection
    # -------------------------------
    results = model(img_bgr, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        st.error("❌ No number plate detected")
    else:
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        plate = img_bgr[y1:y2, x1:x2]

        text, enhanced = read_number_plate(plate)

        # Draw box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.subheader("🔍 Detected Plate")
        st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("✨ Enhanced for OCR")
        st.image(enhanced, clamp=True)

        st.subheader("📄 Extracted Text")
        st.success(text if text else "Could not read text")

        st.subheader("🖼️ Final Detection")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
