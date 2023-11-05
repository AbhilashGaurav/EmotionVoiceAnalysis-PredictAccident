import streamlit as st
import cv2
from deepface import DeepFace

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Face Emotion Detection")

# Create a placeholder for the live video stream
video_placeholder = st.empty()

# Create a flag to control the loop
stop_flag = False

cap = cv2.VideoCapture(1)  # Use camera 1, adjust as needed

while not stop_flag:
    ret, frame = cap.read()

    # Analyze the frame for emotion using DeepFace
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]["dominant_emotion"]
    txt = f"Emotion: {emotion}"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Display the frame and emotion text
    video_placeholder.image(frame, channels="BGR", caption=txt, use_column_width=True)

    # Check if the 'q' key is pressed to stop the loop
    if cv2.waitKey(1) & 0xff == ord('q'):
        stop_flag = True

cap.release()
cv2.destroyAllWindows()
