import numpy as np
import cv2
import face_recognition as fr

imgBase = fr.load_image_file('ImagesBasic/Elon1.jpg')
imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)

imgTest = fr.load_image_file('ImagesBasic/Elon2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgBase)[0]
encodeBase = fr.face_encodings(imgBase)[0]
cv2.rectangle(imgBase, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = fr.face_locations(imgTest)[0]
encodeTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = fr.compare_faces([encodeBase], encodeTest)
faceDist = fr.face_distance([encodeBase], encodeTest)
print(results, faceDist)
cv2.putText(imgTest, f'{results} {round(faceDist[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
cv2.imshow("test1", imgTest)
cv2.waitKey(0)

