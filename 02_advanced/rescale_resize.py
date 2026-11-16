"""Redimensionando imagens, vídeos e (opcionalmente) a resolução da webcam."""
from pathlib import Path

import cv2 as cv

RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def rescale_frame(frame, scale=0.75):
    """Funciona para imagens, vídeos e vídeo ao vivo."""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    # INTER_AREA é a melhor opção para reduzir imagens
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def change_res(capture, width, height):
    """Só funciona com vídeo ao vivo (webcam), não com arquivos: cv.VideoCapture(0)."""
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)


# Lendo um vídeo
capture = cv.VideoCapture(str(RESOURCES / "videos" / "dog.mp4"))
if not capture.isOpened():
    raise SystemExit("Não foi possível abrir o vídeo.")

while True:
    is_true, frame = capture.read()
    if not is_true:  # fim do vídeo
        break

    frame_resized = rescale_frame(frame, scale=0.2)

    cv.imshow("Video", frame)
    cv.imshow("Video redimensionado", frame_resized)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()
