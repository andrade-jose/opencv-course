"""Detecção de contornos."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "cats.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Gatos", img)

# Imagem preta do mesmo tamanho, onde vamos desenhar os contornos
blank = np.zeros(img.shape, dtype="uint8")
cv.imshow("Em branco", blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Cinza", gray)

blur = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow("Blur", blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow("Bordas (Canny)", canny)

# Alternativa ao Canny: binarizar a imagem
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow("Threshold", thresh)

# Em OpenCV 4 e 5 findContours devolve (contornos, hierarquia)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contorno(s) encontrado(s)!")

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)  # cor em BGR: vermelho
cv.imshow("Contornos desenhados", blank)

cv.waitKey(0)
cv.destroyAllWindows()
