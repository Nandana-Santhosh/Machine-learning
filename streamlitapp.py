import streamlit as st
import io
import cv2
import numpy as np
import tensorflow as tf

# Function to load the Keras model and labels
def load_model_and_labels(model_path, labels_path):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return model, class_names

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Streamlit app
def main():
    st.title("Lettuce spinach classifier")
    st.sidebar.header("Settings")

    # Upload an image
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        # Load the model and labels
        model, class_names = load_model_and_labels("keras_model.h5", "labels.txt")

        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the prediction and confidence score
        st.subheader("Prediction:")
        st.write(f"Class: {class_name}")
        st.write(f"Confidence Score: {np.round(confidence_score * 100, 2)}%")

if __name__ == "__main__":
    main()
