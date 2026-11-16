"""Transformações de imagem: translação, rotação, resize, flip e crop."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "park.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Parque", img)


# Translação
def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])  # (largura, altura)
    return cv.warpAffine(img, trans_mat, dimensions)


# -x --> Esquerda
# -y --> Cima
#  x --> Direita
#  y --> Baixo
translated = translate(img, -100, 100)
cv.imshow("Transladada", translated)


# Rotação
def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    if rot_point is None:
        rot_point = (width // 2, height // 2)

    rot_mat = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rot_mat, dimensions)


rotated = rotate(img, -45)
cv.imshow("Rotacionada", rotated)

rotated_rotated = rotate(rotated, -45)  # rotacionar a já rotacionada
cv.imshow("Rotacionada 2x", rotated_rotated)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Redimensionada", resized)

# Flip: 0 = vertical, 1 = horizontal, -1 = ambos
flip = cv.flip(img, -1)
cv.imshow("Flip", flip)

# Crop
cropped = img[200:400, 300:400]
cv.imshow("Recortada", cropped)

cv.waitKey(0)
cv.destroyAllWindows()
