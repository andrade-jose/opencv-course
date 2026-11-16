"""Threshold: separando objeto e fundo (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
mask = gray > 150                              # NumPy: matriz de True/False
binary = (mask * 255).astype("uint8")          # True -> 255, False -> 0
_, binary2 = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
print(np.array_equal(binary, binary2))         # True
adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv.THRESH_BINARY_INV, 11, 9)
cv.imshow("gray > 150", binary)
cv.imshow("adaptativo", adapt)

cv.waitKey(0)
cv.destroyAllWindows()
