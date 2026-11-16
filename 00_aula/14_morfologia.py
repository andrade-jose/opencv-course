"""Dilatar e erodir: limpando máscaras (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
canny = cv.Canny(cv.GaussianBlur(gray, (7, 7), 0), 125, 175)
kernel = np.ones((7, 7), np.uint8)               # kernel = matriz, não tupla!
dilated = cv.dilate(canny, kernel, iterations=3)
eroded = cv.erode(dilated, kernel, iterations=3)
cv.imshow("canny", canny)
cv.imshow("dilatada", dilated)
cv.imshow("erodida", eroded)

cv.waitKey(0)
cv.destroyAllWindows()
