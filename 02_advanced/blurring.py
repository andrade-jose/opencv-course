"""Técnicas de blur (suavização)."""
from pathlib import Path

import cv2 as cv

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "cats.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Gatos", img)

# Média
average = cv.blur(img, (3, 3))
cv.imshow("Blur por media", average)

# Gaussiano
gauss = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow("Blur gaussiano", gauss)

# Mediana (ótimo contra ruído "sal e pimenta")
median = cv.medianBlur(img, 3)
cv.imshow("Blur por mediana", median)

# Bilateral (suaviza preservando as bordas)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow("Bilateral", bilateral)

cv.waitKey(0)
cv.destroyAllWindows()
