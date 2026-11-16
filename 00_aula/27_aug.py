"""Data augmentation: nossas transformações (código da aula, para os alunos digitarem).

Pode ser executado de qualquer pasta: os caminhos partem da raiz do repositório.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")      # matriz (altura, largura, 3) em BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # matriz (altura, largura)
import keras
from keras import layers
aug = keras.Sequential([
    layers.RandomFlip("horizontal"),   # img[:, ::-1]
    layers.RandomRotation(0.05),       # warpAffine
    layers.RandomZoom(0.1),            # resize + recorte
])
foto = img[None, :, :, ::-1] / 255.0   # (1, h, w, 3) RGB
foto = foto.astype("float32")
for i in range(2):
    out = aug(foto, training=True)[0].numpy()
    cv.imshow(f"variacao {i + 1}",
              (out[:, :, ::-1] * 255).astype("uint8"))

cv.waitKey(0)
cv.destroyAllWindows()
