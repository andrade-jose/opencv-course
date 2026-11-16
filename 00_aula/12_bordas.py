"""Bordas: Sobel e Canny (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
ky = kx.T                                        # Sobel Y = transposta!
gx = cv.filter2D(gray, cv.CV_32F, kx)            # borda vertical
gy = cv.filter2D(gray, cv.CV_32F, ky)            # borda horizontal
edges = cv.convertScaleAbs(cv.magnitude(gx, gy)) # intensidade da borda
canny = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 125, 175)
cv.imshow("sobel X", cv.convertScaleAbs(gx))
cv.imshow("sobel X+Y", edges)
cv.imshow("canny", canny)

cv.waitKey(0)
cv.destroyAllWindows()
