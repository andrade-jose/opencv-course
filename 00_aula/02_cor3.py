"""Cor = 3 matrizes empilhadas (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
b, g, r = cv.split(img)                 # 3 matrizes (altura, largura)
print(img[100, 300])                     # [B G R] de um pixel
print(b.shape, g.shape, r.shape)
blank = np.zeros(b.shape, dtype="uint8")
cv.imshow("so azul", cv.merge([b, blank, blank]))
cv.imshow("so verde", cv.merge([blank, g, blank]))
cv.imshow("so vermelho", cv.merge([blank, blank, r]))

cv.waitKey(0)
cv.destroyAllWindows()
