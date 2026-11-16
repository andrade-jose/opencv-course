"""Espaços de cor: BGR, Grayscale, HSV, L*a*b* e RGB."""
from pathlib import Path

import cv2 as cv

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "park.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Parque", img)

# O OpenCV lê imagens em BGR; o matplotlib espera RGB.
# Por isso, plt.imshow(img) mostraria as cores trocadas:
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

# BGR --> Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Cinza", gray)

# BGR --> HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

# BGR --> L*a*b*
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("LAB", lab)

# BGR --> RGB (o imshow do OpenCV espera BGR, então as cores aparecem trocadas)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB (cores trocadas no imshow)", rgb)

# L*a*b* --> BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow("LAB --> BGR", lab_bgr)

cv.waitKey(0)
cv.destroyAllWindows()
