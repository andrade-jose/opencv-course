"""Redimensionar: tamanho fixo para a CNN (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
small = cv.resize(img, (80, 80), interpolation=cv.INTER_AREA)
print(img.shape, "->", small.shape)     # (427, 640, 3) -> (80, 80, 3)
# cv.resize usa (LARGURA, ALTURA); o shape é (altura, largura)!
half = cv.resize(img, None, fx=0.5, fy=0.5)     # mantém a proporção
print(half.shape)
big = cv.resize(small, (640, 427), interpolation=cv.INTER_NEAREST)
cv.imshow("original 640x427", img)
cv.imshow("80x80 (ampliada p/ ver os pixels)", big)

cv.waitKey(0)
cv.destroyAllWindows()
