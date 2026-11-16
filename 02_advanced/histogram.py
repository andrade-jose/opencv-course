"""Cálculo de histogramas (cinza e colorido) com máscara."""
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "cats.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Gatos", img)

blank = np.zeros(img.shape[:2], dtype="uint8")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Cinza", gray)

# Máscara circular no centro da imagem
mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow("Mascara", masked)

# Histograma em escala de cinza
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
#
# plt.figure()
# plt.title("Histograma em cinza")
# plt.xlabel("Bins")
# plt.ylabel("# de pixels")
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

# Histograma colorido (um por canal: B, G, R)
plt.figure()
plt.title("Histograma colorido")
plt.xlabel("Bins")
plt.ylabel("# de pixels")
colors = ("b", "g", "r")
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
