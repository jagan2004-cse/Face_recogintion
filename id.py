import face_recognition 
import cv2 
import numpy as np 
import csv
import os
import glob
from datetime import datetime
video =cv2.VideoCapture(0)
dhoni_image = face_recognition.load_image_file("photos/dhoniai.jpeg")
dhoni_encoding = face_recognition.face_encodings(dhoni_image)[0]
sundar_image = face_recognition.load_image_file("photos/sundar.jpeg")

sundar_encoding = face_recognition.face_encodings(sundar_image)[0]
jagan_image = face_recognition.load_image_file("photos/jagaface.jpg")
jagan_encoding = face_recognition.face_encodings(jagan_image)[0]
 


 
known_face_encoding = [
dhoni_encoding,sundar_encoding,jagan_encoding

]
 
known_faces_names = [
"dhoni","sundar","jagan"


]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
lnwriter.writerow(['Name', 'Time'])
f.flush() 
 
while True:
    _,frame = video.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2

                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

                if name in students:
                    students.remove(name)
                    now =datetime.now()
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
                    f.flush() 
                    print(f"{name} marked present at {current_time}")
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
f.close()