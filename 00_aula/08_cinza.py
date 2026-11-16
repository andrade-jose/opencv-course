"""Escala de cinza: 3 números viram 1 (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
b, g, r = cv.split(img)
manual = (0.114 * b + 0.587 * g + 0.299 * r).astype("uint8")
print(img.shape, "->", gray.shape)             # (427,640,3) -> (427,640)
print(np.abs(gray.astype(int) - manual).max()) # diferença de no máximo 1
cv.imshow("cinza (OpenCV)", gray)
cv.imshow("cinza (na mao)", manual)

cv.waitKey(0)
cv.destroyAllWindows()
