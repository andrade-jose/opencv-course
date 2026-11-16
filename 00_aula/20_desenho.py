"""Desenhar: anotando resultados (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
canvas = np.zeros((500, 500, 3), dtype="uint8")        # tela preta
cv.rectangle(canvas, (0, 0), (250, 250), (0, 255, 0), thickness=-1)
cv.circle(canvas, (250, 250), 40, (0, 0, 255), thickness=-1)
cv.line(canvas, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv.putText(canvas, "Ola, OpenCV!", (0, 225),
           cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
canvas[400:450, 300:400] = (0, 0, 255)                  # pintar = fatiar a matriz
cv.imshow("desenho", canvas)

cv.waitKey(0)
cv.destroyAllWindows()
