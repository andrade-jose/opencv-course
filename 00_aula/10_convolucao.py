"""Filtro = kernel deslizando na matriz (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
patch = gray[158:161, 329:332].astype(np.float32)      # pedaço 3x3 da matriz
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)             # kernel (Sobel X)
print((patch * kx).sum())                                  # multiplica e soma
print(cv.filter2D(gray, cv.CV_32F, kx)[159, 330])          # o OpenCV faz igual

cv.waitKey(0)
cv.destroyAllWindows()
