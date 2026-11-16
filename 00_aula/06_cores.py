"""Trocando cores: BGR, RGB e canais (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
rgb = img[:, :, ::-1]                        # BGR -> RGB (fatia nos canais)
print(np.array_equal(rgb, cv.cvtColor(img, cv.COLOR_BGR2RGB)))   # True
no_red = img.copy()
no_red[:, :, 2] = 0                          # canal 2 = vermelho: zera
cv.imshow("RGB mostrado como BGR", rgb)      # cores trocadas na tela!
cv.imshow("sem vermelho", no_red)
cv.imshow("original", img)

cv.waitKey(0)
cv.destroyAllWindows()
