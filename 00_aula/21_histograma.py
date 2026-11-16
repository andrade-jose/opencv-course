"""Histograma: o retrato dos valores (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
hist = cv.calcHist([gray], [0], None, [256], [0, 256])
print(hist.shape, int(hist.sum()))     # (256, 1) 273280 = nº de pixels
print("valor mais comum:", int(hist.argmax()))
mask = np.zeros(gray.shape, dtype="uint8")
cv.circle(mask, (320, 200), 100, 255, -1)          # só a região do círculo
h2 = cv.calcHist([gray], [0], mask, [256], [0, 256])
print(int(h2.sum()), "pixels dentro da máscara")

cv.waitKey(0)
cv.destroyAllWindows()
