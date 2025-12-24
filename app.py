import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Number Plate Detection", layout="wide")
st.title("🚗 Car Number Plate Detection")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Plate enhancement (UNCHANGED)
def enhance_plate(plate):
    plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return final

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 🔹 YOLO inference
    results = model(img_np)[0]

    if results.boxes is None or len(results.boxes) == 0:
        st.warning("No number plates detected.")
    else:
        st.subheader(f"Detected Plates: {len(results.boxes)}")

        # 🔹 MULTI-PLATE LOOP (THIS IS THE ONLY CHANGE)
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_crop = img_np[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            enhanced_plate = enhance_plate(plate_crop)

            col1, col2 = st.columns(2)
            with col1:
                st.image(plate_crop, caption=f"Plate {idx+1} (Raw)", use_container_width=True)
            with col2:
                st.image(enhanced_plate, caption=f"Plate {idx+1} (Enhanced)", use_container_width=True)
