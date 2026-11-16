"""Toda imagem é uma matriz (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
print(gray.shape)                 # (427, 640): linhas x colunas
print(gray[290, 343])             # UM pixel = UM número (0 a 255)
print(gray[288:296, 340:348])     # um pedaço 8x8 da matriz (a tabela ao lado)
print(gray.dtype, gray.min(), gray.max())

cv.waitKey(0)
cv.destroyAllWindows()
