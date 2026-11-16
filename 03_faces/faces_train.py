"""Treina o reconhecedor facial LBPH com as fotos de resources/faces/train.

Requer o pacote `opencv-contrib-python` (o módulo `cv.face` não vem no `opencv-python`).
Uso:  python faces_train.py
"""
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "resources" / "faces" / "train"
MODEL_PATH = HERE / "face_trained.yml.gz"  # o OpenCV comprime automaticamente com .gz

# Cada subpasta de `train` é uma pessoa. A ordem alfabética define o rótulo numérico.
people = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())

haar_cascade = cv.CascadeClassifier(str(HERE / "haar_face.xml"))
if haar_cascade.empty():
    raise SystemExit("Não foi possível carregar haar_face.xml.")

features = []
labels = []


def create_train():
    for label, person in enumerate(people):
        path = TRAIN_DIR / person

        for img_path in sorted(path.iterdir()):
            img_array = cv.imread(str(img_path))
            if img_array is None:  # arquivo que não é imagem
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)


if __name__ == "__main__":
    if not hasattr(cv, "face"):
        raise SystemExit(
            "Módulo cv.face não encontrado. Instale: pip uninstall opencv-python && "
            "pip install opencv-contrib-python"
        )

    create_train()
    print(f"Treino concluído: {len(features)} rostos de {len(people)} pessoas")

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Treina o reconhecedor com a lista de rostos e a lista de rótulos
    face_recognizer.train(features, np.array(labels))

    face_recognizer.save(str(MODEL_PATH))
    print(f"Modelo salvo em {MODEL_PATH.name}")
