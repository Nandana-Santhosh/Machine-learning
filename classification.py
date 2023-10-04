from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import os
import tensorflow

ctime = 0
ptime = 0

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# model = load_model("keras_Model.h5", compile=False)

from tensorflow.keras.models import load_model

# Load the model
model = load_model("keras_Model.h5", compile=True)

# Compile the model
model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['accuracy'])

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)


while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = image.shape

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image = cv2.flip(image, 1)
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  


    # cv2.rectangle(image,(20,350),(90,440),(0,255,204),cv2.FILLED)
    # cv2.rectangle(image,(20,350),(90,440),(0,0,0),5)


    ctime = time.time()
    fps=1/(ctime-ptime)
    ptime=ctime


    cv2.putText(image,f'FPS:{str(int(fps))}',(0,12),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)


    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    # keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    # if keyboard_input == 27:
    #     break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()




# ##GPT###


# import cv2
# import numpy as np
# from keras.models import load_model
# import time

# ctime = 0
# ptime = 0

# # Load the pre-trained Keras model
# model = load_model('keras_Model.h5')


# from tensorflow.keras.models import load_model

# # Load the model
# model = load_model("keras_Model.h5")

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Load the class labels (if available)
# class_labels = ('labels.txt', 'r')  # Replace with your own class labels

# # Open the webcam
# cap = cv2.VideoCapture(0)


# while True:
#     ret, frame = cap.read()

#     frame = cv2.flip(frame, 1)

#     if not ret:
#         break

#     # Preprocess the frame for object detection
#     input_frame = cv2.resize(frame, (224, 224))
#     input_frame = np.expand_dims(input_frame, axis=0)
#     input_frame = input_frame / 255.0  # Normalize pixel values

#     # Perform object detection
#     predictions = model.predict(input_frame)
#     class_index = np.argmax(predictions)
#     class_label = class_labels[class_index]
#     confidence = predictions[0][class_index]


#     ctime = time.time()
#     fps=1/(ctime-ptime)
#     ptime=ctime


#     cv2.putText(frame,f'FPS:{str(int(fps))}',(0,12),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)


#     # Draw bounding box and label on the frame
#     if confidence > 0.5:  # You can adjust this threshold
#         label = f'{class_label}: {confidence:.2f}'
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow('Object Detection', frame)

#     # Break the loop when the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()




###GPT _ 2 ###


# import cv2
# import numpy as np
# import time
# import tensorflow

# # ...
# ctime = 0
# ptime = 0

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# from tensorflow.keras.models import load_model

# # Load the model
# model = load_model("keras_Model.h5", compile=True)

# # Compile the model
# model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['accuracy'])

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(0)


# while True:
#     ret, image = camera.read()

#     print(ret)

#     print("Image shape:", image.shape)


#     # image = cv2.resize(image, None, fx=0.4, fy=0.4)
#     height, width, channels = image.shape
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#     image = cv2.flip(image, 1)
#     imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

#     ctime = time.time()
#     fps = 1 / (ctime - ptime)
#     ptime = ctime

#     cv2.putText(image, f'FPS:{str(int(fps))}', (0, 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

#     # Make the image a numpy array and reshape it to the model's input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(224, 224, 3)
#     image = (image / 127.5) - 1

#     # Make the image a numpy array
#     # image = np.asarray(image, dtype=np.float32)

# # Normalize the image array
#     # image = (image/ 127.5) - 1

# # Expand the dimensions to match the model's input shape
#     image = np.expand_dims(image, axis=0)


#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Draw bounding box around the detected object
#     if confidence_score > 0.5:  # Adjust the confidence threshold as needed
#         x1, y1, x2, y2 = 50, 50, 150, 150  # Replace with actual bounding box coordinates
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, f"{class_name[2:]}: {confidence_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow("Webcam Image", image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()

