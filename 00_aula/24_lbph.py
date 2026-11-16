"""Reconhecer rostos (LBPH) (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
import os
people = sorted(e.name for e in os.scandir(
    "resources/faces/train") if e.is_dir())
haar = cv.CascadeClassifier("03_faces/haar_face.xml")
rec = cv.face.LBPHFaceRecognizer_create()
rec.read("03_faces/face_trained.yml.gz")   # já treinado
for nome in ("elton_john", "madonna"):
    foto = cv.imread(f"resources/faces/val/{nome}/1.jpg")
    g = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
    x, y, w, h = haar.detectMultiScale(g, 1.1, 4)[0]
    label, dist = rec.predict(g[y:y + h, x:x + w])
    print(nome, "->", people[label], round(dist, 1))
    cv.rectangle(foto, (x, y), (x + w, y + h),
                 (0, 255, 0), 3)
    cv.imshow(f"{nome} -> {people[label]}", foto)

cv.waitKey(0)
cv.destroyAllWindows()
