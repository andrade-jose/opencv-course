"""Contornos e caixas: achando objetos (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
cats = cv.imread("resources/photos/cats.jpg")
g = cv.cvtColor(cats, cv.COLOR_BGR2GRAY)
edges = cv.Canny(cv.GaussianBlur(g, (5, 5), 0), 125, 175)
contours, _ = cv.findContours(
    edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
boxes = [cv.boundingRect(c) for c in contours]   # x,y,w,h
boxes = [b for b in boxes if b[2] * b[3] > 2000]
print(len(contours), "contornos,", len(boxes), "caixas")
out = cats.copy()
for (x, y, w, h) in boxes:
    cv.rectangle(out, (x, y), (x + w, y + h),
                 (0, 255, 0), 2)
    roi = cats[y:y + h, x:x + w]     # recorte p/ a CNN
cv.imshow("caixas dos contornos grandes", out)

cv.waitKey(0)
cv.destroyAllWindows()
