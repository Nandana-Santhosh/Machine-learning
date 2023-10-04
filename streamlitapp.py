import streamlit as st
import tensorflow as tf
import numpy as np
import time
import os
os.environ["DISPLAY"] = ":0"  # Set the DISPLAY environment variable
import cv2


# Load your TensorFlow model and labels here
model = tf.keras.models.load_model("keras_model.h5")
class_names = open("labels.txt", "r").readlines()

# Initialize webcam
camera = cv2.VideoCapture(0)

# Function to perform predictions
def predict(image):
    # Preprocess the image as needed

    # Resize the image to match the model's input shape
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict using the loaded model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Streamlit app title
st.title("Lettucespinach model")

# Add a Start button to initiate webcam feed
start_button = st.button("Start")

if start_button:
    num_inputs = 5  # Define the number of inputs to process
    input_counter = 0  # Initialize the input counter

    while input_counter < num_inputs:
        ret, frame = camera.read()
        if not ret:
            st.text("Error: Unable to capture webcam feed.")
            break

        # Perform prediction on the frame
        class_name, confidence_score = predict(frame)

        # Display the webcam feed
        st.image(frame, caption="Webcam Image", use_column_width=True)

        # Display prediction results
        st.subheader("Prediction Results:")
        st.text(f"Class: {class_name[2:]}")
        st.text(f"Confidence Score: {np.round(confidence_score * 100, 2)}%")

        input_counter += 1  # Increment the input counter

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV window
camera.release()
cv2.destroyAllWindows()
