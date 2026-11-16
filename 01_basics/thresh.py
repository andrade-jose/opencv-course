"""Thresholding (binarização) simples e adaptativo."""
from pathlib import Path

import cv2 as cv

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "cats.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Gatos", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Cinza", gray)

# Threshold simples: pixel > 150 vira 255, senão 0
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow("Threshold simples", thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Threshold simples invertido", thresh_inv)

# Threshold adaptativo: o limiar é calculado por vizinhança (11x11), menos constante C=9
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9
)
cv.imshow("Threshold adaptativo", adaptive_thresh)

cv.waitKey(0)
cv.destroyAllWindows()
