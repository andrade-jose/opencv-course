"""5 funções essenciais do OpenCV: cinza, blur, bordas, dilatação/erosão, resize e crop."""
from pathlib import Path

import cv2 as cv
import numpy as np

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

img = cv.imread(str(RESOURCES / "photos" / "park.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Parque", img)

# Convertendo para escala de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Cinza", gray)

# Blur gaussiano (sigmaX=0 -> o OpenCV calcula o sigma a partir do tamanho do kernel)
blur = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow("Blur", blur)

# Detecção de bordas (Canny)
canny = cv.Canny(blur, 125, 175)
cv.imshow("Bordas (Canny)", canny)

# Dilatando a imagem
# O kernel precisa ser uma matriz (e não uma tupla): np.ones((7, 7), np.uint8)
kernel = np.ones((7, 7), np.uint8)
dilated = cv.dilate(canny, kernel, iterations=3)
cv.imshow("Dilatada", dilated)

# Erodindo
eroded = cv.erode(dilated, kernel, iterations=3)
cv.imshow("Erodida", eroded)

# Resize (largura, altura)
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Redimensionada", resized)

# Crop: imagens são arrays NumPy -> basta fatiar [linhas, colunas]
cropped = img[50:200, 200:400]
cv.imshow("Recortada", cropped)

cv.waitKey(0)
cv.destroyAllWindows()
