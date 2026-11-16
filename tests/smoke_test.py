"""Smoke test: executa todos os scripts da aula (00_aula, exceto Keras) e das seções 1 a 3 SEM abrir janelas e confere que não dão erro.

Útil depois de atualizar o OpenCV (`pip install -U opencv-contrib-python`) para saber se algo quebrou.
O capstone (04_capstone) fica de fora porque precisa do TensorFlow e do dataset do Kaggle.

Uso:  python tests/smoke_test.py
"""
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    [*ROOT.glob("01_basics/*.py"), *ROOT.glob("02_advanced/*.py")]
    + [ROOT / "03_faces" / "face_detect.py", ROOT / "03_faces" / "face_recognition.py"]
    # gabaritos da aula; os de Keras (cnn, aug) só rodam com TensorFlow instalado
    + [p for p in ROOT.glob("00_aula/*.py") if not p.stem.endswith(("_cnn", "_aug"))]
)


def run_headless(script):
    """Roda um script com imshow/waitKey desativados e o matplotlib sem interface gráfica."""
    import matplotlib

    matplotlib.use("Agg")
    import cv2

    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: ord("d")  # simula a tecla "d" (sai dos loops de vídeo)
    cv2.destroyAllWindows = lambda *args, **kwargs: None

    sys.argv = [script]
    runpy.run_path(script, run_name="__main__")


def main():
    failures = []
    for script in SCRIPTS:
        # cwd temporário: os scripts devem funcionar de qualquer diretório
        result = subprocess.run(
            [sys.executable, __file__, "--run", str(script)],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
        )
        status = "OK  " if result.returncode == 0 else "FAIL"
        print(f"{status} {script.relative_to(ROOT)}")
        if result.returncode != 0:
            failures.append(script)
            print(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "")

    print(f"\n{len(SCRIPTS) - len(failures)}/{len(SCRIPTS)} scripts passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        run_headless(sys.argv[2])
    else:
        sys.exit(main())
