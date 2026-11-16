"""Indexação e recorte: [linha, coluna] (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
crop = img[50:200, 200:400]            # linhas 50-200, colunas 200-400
print(crop.shape)                       # (150, 200, 3)
paint = img.copy()                      # copy: senão altera a original!
paint[100:200, 300:400] = (0, 0, 255)   # pinta de vermelho (BGR)
cv.imshow("recorte", crop)
cv.imshow("pintada", paint)

cv.waitKey(0)
cv.destroyAllWindows()
