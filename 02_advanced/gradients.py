"""Gradientes e detecção de bordas: Laplacian, Sobel e Canny."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "park.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Parque", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Cinza", gray)

# Laplacian (CV_64F porque o gradiente pode ser negativo)
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

# Sobel nas direções x e y
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

# Valor absoluto + conversão para uint8. Atenção: bitwise_or em float64 NÃO dá erro,
# mas opera nos bits dos números e devolve uma imagem sem sentido.
sobelx = cv.convertScaleAbs(sobelx)
sobely = cv.convertScaleAbs(sobely)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow("Sobel X", sobelx)
cv.imshow("Sobel Y", sobely)
cv.imshow("Sobel combinado", combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow("Canny", canny)

cv.waitKey(0)
cv.destroyAllWindows()
