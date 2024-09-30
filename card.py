import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load face images and encodings
dhoni_image = face_recognition.load_image_file("photos/dhoniai.jpeg")
dhoni_encoding = face_recognition.face_encodings(dhoni_image)[0]

sundar_image = face_recognition.load_image_file("photos/sundar.jpeg")
sundar_encoding = face_recognition.face_encodings(sundar_image)[0]

known_face_encoding = [dhoni_encoding, sundar_encoding]
known_faces_names = ["dhoni", "sundar"]

# Initialize video capture
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10, 100)
            font_scale = 1.5
            font_color = (255, 0, 0)
            thickness = 3
            line_type = 2

            cv2.putText(frame, f"{name} Present", bottom_left_corner, font, font_scale, font_color, thickness, line_type)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

