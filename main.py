import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model_path = "best.pt" # Change this if needed
model = YOLO(model_path)

st.title("Classroom Attendance Tracker 🏫")

# Sidebar for navigation
option = st.sidebar.radio("Chose an option", ["Upload Image", "Live Camera"])

# Function to detect people in an image
def detect_people(frame): 
  results = model(frame) # Run YOLO dectection
  detected_objects = results[0].boxes
  num_students = 0

  for box in detected_objects:
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    conf = box.conf[0].item()
    cls = int(box.cls[0].item())

    if cls == 0:
      num_students += 1
      label = f'Person {conf:.2f}'
      cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 2)
      cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

  return frame, num_students

# Upload Image Option
if option == "Upload Image":
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    processed_image, num_students = detect_people(image)
    st.image(processed_image, caption=f"Detected {num_students} students", use_column_width=True)