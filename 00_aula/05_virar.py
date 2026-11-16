"""Virar e desvirar a imagem (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
v = img[::-1]                    # inverte as linhas  (= cv.flip(img, 0))
h = img[:, ::-1]                 # inverte as colunas (= cv.flip(img, 1))
back = h[:, ::-1]                # desvirar: virar de novo
print(np.array_equal(back, img)) # True: voltou igual, pixel a pixel
r90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
print(np.array_equal(cv.rotate(r90, cv.ROTATE_90_COUNTERCLOCKWISE), img))  # True
cv.imshow("vertical", v)
cv.imshow("horizontal", h)
cv.imshow("rotacionada 90", r90)

cv.waitKey(0)
cv.destroyAllWindows()
