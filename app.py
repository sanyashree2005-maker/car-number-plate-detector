import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Car Number Plate Detection", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- LIGHT DISPLAY ENHANCEMENT ----------------
def enhance_plate_for_display(plate):
    """
    Very light enhancement ONLY for visualization.
    No thresholding, no morphology.
    """
    try:
        import cv2

        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        merged = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced
    except:
        return plate  # fallback (cloud-safe)

# ---------------- UI ----------------
st.title("🚗 Car Number Plate Detection using YOLO")

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- DETECTION ----------------
    results = model(img_np, conf=0.4)

    plates = []
    annotated = img_np.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img_np[y1:y2, x1:x2]

            if plate_crop.size > 0:
                plates.append(plate_crop)

            # draw box
            try:
                import cv2
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )
            except:
                pass

    st.subheader("Detected Plate Regions")
    st.image(annotated, use_container_width=True)

    # ---------------- DISPLAY PLATES ----------------
    if plates:
        st.subheader(f"Detected Plates ({len(plates)})")

        cols = st.columns(2)
        for i, plate in enumerate(plates):
            enhanced = enhance_plate_for_display(plate)

            with cols[i % 2]:
                st.image(plate, caption=f"Plate {i+1} (Raw)")
                st.image(enhanced, caption=f"Plate {i+1} (Enhanced)")
    else:
        st.warning("No plates detected.")

    # ---------------- OCR NOTE ----------------
    ##st.info(
       ## "ℹ OCR is disabled on Streamlit Cloud.\n\n"
        ##"Text extraction works in local or Docker deployments with Tesseract installed."
    )
