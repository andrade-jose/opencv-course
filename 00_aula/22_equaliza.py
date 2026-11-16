"""Equalização: corrigindo contraste (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
dark = cv.convertScaleAbs(gray, alpha=0.4, beta=10)   # simula foto escura
eq = cv.equalizeHist(dark)                            # redistribui os valores
print(dark.min(), dark.max(), "->", eq.min(), eq.max())
cv.imshow("escura", dark)
cv.imshow("equalizada", eq)

cv.waitKey(0)
cv.destroyAllWindows()
