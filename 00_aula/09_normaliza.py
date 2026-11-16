"""Normalizar: de 0–255 para 0–1 (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
x = gray.astype("float32") / 255.0
print(x.dtype, x.min(), x.max())          # float32 0.0 0.937 (o máximo era 239)
back = np.round(x * 255).astype("uint8")
print(np.array_equal(back, gray))         # True: sem perda
x4 = x.reshape(1, *x.shape, 1)            # formato da CNN: (lote, altura, largura, canais)
print(x4.shape)                           # (1, 427, 640, 1)

cv.waitKey(0)
cv.destroyAllWindows()
