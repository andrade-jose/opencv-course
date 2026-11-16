"""Rotacionar e mover (e desfazer) (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
h, w = img.shape[:2]
M = cv.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)    # matriz 2x3: gira 30°
rot = cv.warpAffine(img, M, (w, h))
undo = cv.warpAffine(rot, cv.invertAffineTransform(M), (w, h))   # desvirar: inversa
T = np.float32([[1, 0, 100], [0, 1, 50]])                 # 100 px →, 50 px ↓
moved = cv.warpAffine(img, T, (w, h))
cv.imshow("rotacionada 30", rot)
cv.imshow("desfeita (cantos pretos = perdidos)", undo)
cv.imshow("movida", moved)

cv.waitKey(0)
cv.destroyAllWindows()
