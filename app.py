import streamlit as st
import numpy as np
from PIL import Image
import pytesseract

# ⚠️ DO NOT import cv2 here
# ⚠️ DO NOT import ultralytics here

def load_cv2():
    import cv2
    return cv2

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("best.pt")

model = load_model()

def read_number_plate(plate_img):
    cv2 = load_cv2()

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    config = (
        "--oem 3 "
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

st.set_page_config(page_title="Car Number Plate Detection")

st.title("🚗 Car Number Plate Detection")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)

    results = model(img_np, verbose=False)[0]

    if results.boxes is None:
        st.warning("No number plate detected.")
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            plate_text = read_number_plate(plate)

            st.image(plate, caption="Detected Plate", width=300)
            st.success(f"Detected Text: **{plate_text}**")
