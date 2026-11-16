"""Máscaras: isolando regiões da imagem."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "cats_2.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Gatos", img)

# A máscara precisa ter o mesmo tamanho (altura x largura) da imagem
blank = np.zeros(img.shape[:2], dtype="uint8")
cv.imshow("Imagem em branco", blank)

circle = cv.circle(blank.copy(), (img.shape[1] // 2 + 45, img.shape[0] // 2), 100, 255, -1)

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow("Forma estranha", weird_shape)

masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow("Imagem com mascara", masked)

cv.waitKey(0)
cv.destroyAllWindows()
