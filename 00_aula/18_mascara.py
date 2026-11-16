"""Máscaras: mostrar só o que importa (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
mask = np.zeros(img.shape[:2], dtype="uint8")        # só 2 dimensões!
cv.circle(mask, (320, 200), 120, 255, -1)            # círculo branco no centro
only = cv.bitwise_and(img, img, mask=mask)           # mantém o círculo
hidden = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))   # esconde o círculo
cv.imshow("mascara", mask)
cv.imshow("so o circulo", only)
cv.imshow("circulo escondido", hidden)

cv.waitKey(0)
cv.destroyAllWindows()
