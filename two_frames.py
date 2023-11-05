import streamlit as st
import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
from deepface import DeepFace

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the camera
cap = cv2.VideoCapture(1)  # Use the desired camera (0 or 1)

# Face detection using dlib
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the aspect ratio of the eyes
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

st.title("Drowsiness Detector and Emotion Recognizer")

# Create two columns for video streams
col1, col2 = st.columns(2)

# Create placeholders for the video streams
video_placeholder1 = col1.empty()
video_placeholder2 = col2.empty()

while True:
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Drowsiness detection
    faces = face_detector(gray_scale)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2

        Eye_Rat = round(Eye_Rat, 2)

        if Eye_Rat < 0.15:
            st.warning("DROWSINESS DETECTED")
            st.warning("Alert!!!! WAKE UP DUDE")
            engine.say("Alert!!!! WAKE UP DUDE")
            engine.runAndWait()

        video_placeholder1.image(frame, channels="BGR")

    # Emotion recognition
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]["dominant_emotion"]
    txt = f"Emotion: {emotion}"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    video_placeholder2.image(frame, channels="BGR", caption=txt)

    # Check if the 'q' key is pressed to stop the loop
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
