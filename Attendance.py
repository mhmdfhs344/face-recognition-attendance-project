import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
className = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])


def encodings(nface):
    encodelist = []
    for img in nface:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodelist.append(encode)
    print("encoding is completed")
    return encodelist


def attendance(person):
    with open('attendance.csv', 'r+') as f:
        data = f.readlines()
        oldNames = []
        for line in data:
            entry = line.split(',')
            oldNames.append(entry[0])
        if person not in oldNames:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{person},{time}')
    f.close()


encodeListBase = encodings(images)
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodeListBase, encodeFace)
        faceDis = fr.face_distance(encodeListBase, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()

            # print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # face rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # name rectangle
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # Text name
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('WebCam', img)
    cv2.waitKey(1)



