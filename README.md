# 🚗 Car Number Plate Detection using YOLO

This project is a computer vision application designed to detect vehicle number plates from images using a YOLO-based object detection model. The application is deployed as a web interface using Streamlit, allowing users to easily upload images and view detected number plate regions.

---

## 📌 Problem Statement
Manual identification of vehicle number plates from images is time-consuming and error-prone, especially in surveillance and traffic monitoring systems. An automated and reliable solution is required to accurately detect number plates from images containing one or multiple vehicles.

---

## 🎯 Objective
- To build an automated system that detects car number plates from images
- To support detection of multiple number plates in a single image
- To provide a simple web interface for easy usage and visualization

---

## ⚙️ System Workflow
1. User uploads a car image through the Streamlit interface  
2. The YOLO model processes the image and detects number plate regions  
3. Bounding boxes are drawn around detected plates  
4. Each detected plate is cropped and displayed separately  

---

## ✨ Key Features
- YOLO-based real-time object detection
- Supports multiple number plate detection
- Fast and accurate inference
- User-friendly Streamlit web interface
- Visual comparison of detected plate regions

---

## 🛠️ Technologies Used
- **Python** – Core programming language  
- **YOLO (Ultralytics)** – Object detection model  
- **OpenCV** – Image processing  
- **NumPy** – Array and image operations  
- **Streamlit** – Web application framework  

---

## 🚀 Deployment
The application is deployed on **Streamlit Cloud**.

> ⚠️ **Note:**  
OCR (text extraction from number plates) is disabled in the Streamlit Cloud deployment due to system limitations.  
The current deployment focuses only on **number plate detection**, not text recognition.

---

## 📥 How to Run Locally
1. Clone the repository  
2. Install required dependencies from `requirements.txt`  
3. Run the app using:
   ```bash
   streamlit run app.py
