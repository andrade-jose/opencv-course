"""Bitwise: lógica pixel a pixel (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
blank = np.zeros((400, 400), dtype="uint8")
rect = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circ = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow("AND (interseção)", cv.bitwise_and(rect, circ))
cv.imshow("OR (união)", cv.bitwise_or(rect, circ))
cv.imshow("XOR (so o que nao se sobrepoe)", cv.bitwise_xor(rect, circ))
cv.imshow("NOT (inverte)", cv.bitwise_not(circ))

cv.waitKey(0)
cv.destroyAllWindows()
