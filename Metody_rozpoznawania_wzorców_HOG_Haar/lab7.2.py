import cv2
import numpy as np

replacement_img = cv2.imread("zad8_a.png")
replacement_img = cv2.cvtColor(replacement_img, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nie można uruchomić kamery.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        scaled_replacement = cv2.resize(replacement_img, (w, h))

        if scaled_replacement.shape[2] == 3:
            frame[y:y+h, x:x+w] = cv2.cvtColor(scaled_replacement, cv2.COLOR_RGB2BGR)

    cv2.imshow("Rozpoznawanie i podmiana twarzy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
