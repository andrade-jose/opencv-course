"""Aritmética: brilho, estouro e negativo (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
bright = cv.add(img, 60)            # soma com saturação: para em 255
wrong = img + 60                     # NumPy: passou de 255, volta a 0!
inverted = 255 - img                 # negativo (inverte as cores)
contrast = cv.convertScaleAbs(img, alpha=1.5, beta=0)   # alpha = contraste
cv.imshow("original", img)
cv.imshow("cv.add(img, 60)", bright)
cv.imshow("img + 60 (estouro)", wrong)
cv.imshow("255 - img (negativo)", inverted)

cv.waitKey(0)
cv.destroyAllWindows()
