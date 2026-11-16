"""Detectar rostos (Haar Cascade) (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
group = cv.imread("resources/photos/group_2.jpg")
g = cv.cvtColor(group, cv.COLOR_BGR2GRAY)              # Haar trabalha em cinza
haar = cv.CascadeClassifier("03_faces/haar_face.xml")
faces = haar.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4)
print(len(faces), "caixas")
for (x, y, w, h) in faces:
    cv.rectangle(group, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv.imshow("rostos detectados", group)

cv.waitKey(0)
cv.destroyAllWindows()
