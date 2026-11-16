"""Detecção de rostos com Haar Cascades."""
from pathlib import Path

import cv2 as cv

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "resources"

img = cv.imread(str(RESOURCES / "photos" / "group_2.jpg"))
if img is None:
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")
cv.imshow("Grupo de 5 pessoas", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Pessoas em cinza", gray)

haar_cascade = cv.CascadeClassifier(str(HERE / "haar_face.xml"))
if haar_cascade.empty():
    raise SystemExit("Não foi possível carregar haar_face.xml.")

# minNeighbors alto = menos falsos positivos (mas pode perder rostos).
# Experimente 1 para ver muitos "rostos fantasmas" e 8 para ver rostos sendo perdidos.
# Mesmo com 4 aparecem 1-2 falsos positivos: é um limite dos Haar Cascades (assunto para
# comparar com detectores modernos baseados em redes neurais).
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f"Número de rostos encontrados = {len(faces_rect)}")

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Rostos detectados", img)

cv.waitKey(0)
cv.destroyAllWindows()
