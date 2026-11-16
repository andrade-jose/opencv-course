"""Blur: tirando ruído (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
k = np.ones((15, 15), np.float32) / 225      # kernel de média 15x15
manual = cv.filter2D(img, -1, k)             # convolução "na mao"
blur = cv.blur(img, (15, 15))                # mesma coisa, pronta
print(np.abs(manual.astype(int) - blur).max())   # 0
gauss = cv.GaussianBlur(img, (15, 15), 0)    # pesos em forma de sino
med = cv.medianBlur(img, 15)                 # mediana: ótimo p/ ruído "sal e pimenta"
cv.imshow("media (filter2D)", manual)
cv.imshow("gaussiano", gauss)
cv.imshow("mediana", med)

cv.waitKey(0)
cv.destroyAllWindows()
