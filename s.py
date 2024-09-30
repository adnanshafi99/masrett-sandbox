import pickle
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st

# Global variable for predicted word
predicted_word = "No sign detected"  # Initialize or reset the variable

# Load the trained model and label encoder from model.p
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    label_encoder = model_data['label_encoder']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Streamlit web app layout
#st.title("Real-Time Sign Language Detection")
#st.write("This application detects sign language in real time using your webcam.")

# Placeholder for the video feed
frame_placeholder = st.empty()

# Start webcam for real-time detection
cap = cv2.VideoCapture(0)

# Loop for capturing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame and extract landmarks
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).reshape(1, 21, 3)

            # Flatten the landmarks for prediction
            landmarks_flat = landmarks.reshape(1, -1)  # Shape: (1, 63)

            # Predict the sign language word
            predictions = model.predict(landmarks_flat)
            confidence = np.max(model.predict_proba(landmarks_flat))  # Confidence level
            predicted_class = label_encoder.inverse_transform([predictions[0]])

            if confidence >= 0.40:  # Only show if confidence is 40% or more
                predicted_word = predicted_class[0]  # Update the predicted word

                # Extract the bounding box coordinates
                h, w, _ = frame.shape  # Get dimensions of the frame
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Put the predicted word on the frame
                cv2.putText(frame, f"{predicted_word} ({confidence:.2f})", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the frame on Streamlit
    frame_placeholder.image(frame, channels="BGR")

# Clean up
cap.release()
