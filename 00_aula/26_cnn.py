"""O que a CNN faz com a matriz (código da aula, para os alunos digitarem).

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
model = keras.Sequential([
    keras.Input(shape=(80, 80, 1)),     # nossa matriz
    layers.Conv2D(8, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),         # reduz pela metade
    layers.Flatten(),                    # matriz -> vetor
    layers.Dense(2, activation="softmax"),
])
model.summary()
print(model.output_shape, model.count_params())

cv.waitKey(0)
cv.destroyAllWindows()
