"""Lendo imagens e vídeos com OpenCV."""
from pathlib import Path

import cv2 as cv

# Pasta "resources" na raiz do projeto (funciona de qualquer diretório de execução)
RESOURCES = Path(__file__).resolve().parents[1] / "resources"

# --- Lendo uma imagem ---
img = cv.imread(str(RESOURCES / "photos" / "cats.jpg"))
if img is None:  # imread NÃO levanta erro se o arquivo não existe: devolve None
    raise SystemExit("Imagem não encontrada. Verifique o caminho.")

cv.imshow("Gatos", img)
cv.waitKey(0)  # espera qualquer tecla

# --- Lendo um vídeo ---
capture = cv.VideoCapture(str(RESOURCES / "videos" / "dog.mp4"))
if not capture.isOpened():
    raise SystemExit("Não foi possível abrir o vídeo.")

while True:
    is_true, frame = capture.read()

    # Quando `is_true` é False o vídeo acabou (ou o frame não pôde ser lido).
    if not is_true:
        break

    cv.imshow("Video", frame)
    # Pressione "d" para sair antes do fim
    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()
