"""Desenhando formas e escrevendo texto em imagens."""
import cv2 as cv
import numpy as np

# Tela preta 500x500 com 3 canais (BGR)
blank = np.zeros((500, 500, 3), dtype="uint8")
cv.imshow("Em branco", blank)

# 1. Pintar uma região da imagem (as cores no OpenCV são BGR, não RGB!)
blank[200:300, 300:400] = 0, 0, 255
cv.imshow("Vermelho", blank)

# 2. Retângulo (thickness=-1 preenche a forma)
cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 255, 0), thickness=-1)
cv.imshow("Retangulo", blank)

# 3. Círculo
cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 40, (0, 0, 255), thickness=-1)
cv.imshow("Circulo", blank)

# 4. Linha
cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv.imshow("Linha", blank)

# 5. Texto (as fontes Hershey não suportam acentos, por isso sem "á", "ç" etc.)
cv.putText(blank, "Ola, OpenCV!", (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
cv.imshow("Texto", blank)

cv.waitKey(0)
cv.destroyAllWindows()
