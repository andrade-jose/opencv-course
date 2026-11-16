"""Separando e juntando canais de cor."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "park.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Parque", img)

blank = np.zeros(img.shape[:2], dtype="uint8")

b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow("Azul", blue)
cv.imshow("Verde", green)
cv.imshow("Vermelho", red)

print(img.shape)  # (altura, largura, 3)
print(b.shape)  # (altura, largura)
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r])
cv.imshow("Imagem reunida", merged)

cv.waitKey(0)
cv.destroyAllWindows()
