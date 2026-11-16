"""Preparando os dados para a CNN (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
import os
nomes = ["madonna", "elton_john"]      # capstone: 10
X, y = [], []
for label, nome in enumerate(nomes):
    pasta = f"resources/faces/train/{nome}"
    for f in os.scandir(pasta):
        im = cv.imread(f.path, cv.IMREAD_GRAYSCALE)
        im = cv.resize(im, (80, 80),
                       interpolation=cv.INTER_AREA)
        X.append(im)
        y.append(label)
X = np.array(X, dtype="float32")
X = X.reshape(-1, 80, 80, 1) / 255.0
y = np.array(y)
print(X.shape, y.shape, X.min(), X.max())
amostra = [(X[i, :, :, 0] * 255).astype("uint8")
           for i in (0, 1, 2, 30)]
cv.imshow("amostras 80x80", cv.resize(
    np.hstack(amostra), None, fx=3, fy=3,
    interpolation=cv.INTER_NEAREST))

cv.waitKey(0)
cv.destroyAllWindows()
