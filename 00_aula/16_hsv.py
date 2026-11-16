"""HSV: deixando uma cor aparente (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([8, 120, 120])                  # laranja/amarelo: H de 8 a 30
upper = np.array([30, 255, 255])
mask = cv.inRange(hsv, lower, upper)             # matriz 0/255: onde há a cor
only = cv.bitwise_and(img, img, mask=mask)       # mantém só esses pixels
cv.imshow("mascara da cor", mask)
cv.imshow("so as folhas laranjas", only)

cv.waitKey(0)
cv.destroyAllWindows()
