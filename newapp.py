import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load the model
model = load_model("keras_model.h5")

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create a Streamlit web app
st.title("Webcam Image Classification")

# Function to predict and display the results
def predict(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Access the webcam
cap = cv2.VideoCapture(0)

# Global variable to count captured frames
frame_count = 0

# Main Streamlit loop
while frame_count < 5:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Perform prediction
    class_name, confidence_score = predict(frame)

    # Display the webcam feed and prediction results
    st.image(frame, channels="BGR", use_column_width=True, caption="Webcam Feed")
    st.write(f"Class: {class_name[2:]}")
    st.write(f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]} %")

    # Increment the frame count
    frame_count += 1

# Release the webcam
cap.release()
