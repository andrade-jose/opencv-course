"""Reconhecimento facial com LBPH (requer `opencv-contrib-python`).

Uso:
    python face_recognition.py                              # usa uma foto de exemplo
    python face_recognition.py caminho/para/outra_foto.jpg  # usa a sua foto
"""
import sys
from pathlib import Path

import cv2 as cv

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "resources"
MODEL_PATH = HERE / "face_trained.yml.gz"

if not hasattr(cv, "face"):
    raise SystemExit(
        "Módulo cv.face não encontrado. Instale: pip uninstall opencv-python && "
        "pip install opencv-contrib-python"
    )
if not MODEL_PATH.exists():
    raise SystemExit("Modelo não encontrado. Rode antes: python faces_train.py")

# Mesma ordem usada no treino (subpastas de resources/faces/train em ordem alfabética)
people = sorted(p.name for p in (RESOURCES / "faces" / "train").iterdir() if p.is_dir())

haar_cascade = cv.CascadeClassifier(str(HERE / "haar_face.xml"))

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(str(MODEL_PATH))

img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESOURCES / "faces" / "val" / "elton_john" / "1.jpg"
img = cv.imread(str(img_path))
if img is None:
    raise SystemExit(f"Imagem não encontrada: {img_path}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Pessoa", gray)

# Detecta o rosto na imagem
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]

    # No LBPH, quanto MENOR a confiança (distância), melhor a correspondência
    label, confidence = face_recognizer.predict(faces_roi)
    name = people[label].replace("_", " ").title()
    print(f"Rótulo = {name} com confiança (distância) de {confidence:.1f}")

    cv.putText(img, name, (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Rosto detectado", img)

cv.waitKey(0)
cv.destroyAllWindows()
