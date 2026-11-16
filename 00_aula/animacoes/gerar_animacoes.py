"""Gera vídeos curtos (MP4) mostrando, número por número, o que cada operação faz na matriz da imagem.

Pega um pedaço 8x8 de verdade da foto resources/photos/park.jpg e anima o cálculo em cada pixel,
para mostrar que não é mágica: é conta em cima de uma matriz.

Uso (a partir de qualquer pasta):
    python 00_aula/animacoes/gerar_animacoes.py            # gera todos em 00_aula/animacoes/videos/
    python 00_aula/animacoes/gerar_animacoes.py brilho     # só um (ids no final do arquivo)

Precisa de: opencv-python, numpy, pillow e do ffmpeg instalado (para gravar o MP4).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "videos"
W, H, FPS = 1280, 720, 24

# ---------- visual ----------
BG, INK, MUTED = (247, 248, 250), (27, 32, 41), (110, 118, 130)
ACCENT, WARN, CODEBG = (17, 150, 120), (214, 60, 60), (27, 32, 41)
FD = "/usr/share/fonts/truetype/dejavu/"


def font(name, size):
    return ImageFont.truetype(FD + name, size)


F_TITLE, F_TXT, F_BOLD = font("DejaVuSans-Bold.ttf", 38), font("DejaVuSans.ttf", 24), font("DejaVuSans-Bold.ttf", 24)
F_CODE = font("DejaVuSansMono.ttf", 24)


def lum(c):
    return 0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0] if len(c) == 3 else c


def gray_rgb(v):
    v = int(max(0, min(255, round(v))))
    return (v, v, v)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class Video:
    """Grava frames direto no ffmpeg (H.264, abre em PowerPoint/Keynote/qualquer player)."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg não encontrado: instale-o para gravar os vídeos")
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, img, seconds=0.0):
        data = np.asarray(img.convert("RGB")).tobytes()
        for _ in range(max(1, round(seconds * FPS))):
            self.p.stdin.write(data)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def base(title, code, subtitle=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((50, 28), title, font=F_TITLE, fill=INK)
    if subtitle:
        d.text((50, 80), subtitle, font=F_TXT, fill=MUTED)
    d.rounded_rectangle((50, 122, W - 50, 176), 12, fill=CODEBG)
    d.text((72, 135), code, font=F_CODE, fill=(230, 235, 240))
    return im, d


def draw_cell(d, x, y, s, text, fill, hl=None, fsize=None, text_color=None):
    d.rectangle((x, y, x + s - 2, y + s - 2), fill=fill)
    if hl:
        d.rectangle((x, y, x + s - 2, y + s - 2), outline=hl, width=5)
    fs = fsize or int(s * 0.36)
    f = font("DejaVuSansMono-Bold.ttf", fs)
    col = text_color or ((255, 255, 255) if lum(fill) < 140 else INK)
    w = d.textlength(text, font=f)
    d.text((x + (s - 2 - w) / 2, y + (s - 2) / 2 - fs * 0.62), text, font=f, fill=col)


def wrap_lines(d, lines, x, y, f=F_TXT, gap=38, colors=None):
    for i, t in enumerate(lines):
        d.text((x, y + i * gap), t, font=f, fill=(colors[i] if colors else INK))


def get_data():
    img = cv.imread(str(ROOT / "resources/photos/park.jpg"))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray, gray[288:296, 340:348].copy()


# ---------- animação 1: conta pixel a pixel ----------
def elementwise(name, title, code, P, fn, opfmt, explain, full_fn, gray, notes=None, sat_note=None):
    """Mostra a matriz antes, aplica o cálculo célula por célula e termina com o resultado."""
    v = Video(OUT / f"{name}.mp4")
    Q = fn(P)
    n, S, X0, Y0 = P.shape[0], 60, 60, 192
    cur_text = None

    def frame(k, phase, t=0.0):
        im, d = base(title, code)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                val, fill, hl, txt = P[i, j], gray_rgb(P[i, j]), None, str(int(P[i, j]))
                if idx < k:
                    val = Q[i, j]
                    fill, txt = gray_rgb(val), str(int(val))
                if idx == k and phase == "calc":
                    hl = ACCENT
                    txt = str(int(P[i, j]))
                draw_cell(d, X0 + j * S, Y0 + i * S, S, txt, fill, hl)
        d.text((X0, Y0 + n * S + 14), "matriz (8×8 pixels reais da foto)", font=F_TXT, fill=MUTED)
        rx = 640
        if phase == "calc" and k < n * n:
            i, j = divmod(k, n)
            a, b = int(P[i, j]), int(Q[i, j])
            d.text((rx, 210), f"pixel [{i}, {j}]", font=F_BOLD, fill=ACCENT)
            d.text((rx, 255), opfmt(a, b), font=font("DejaVuSansMono-Bold.ttf", 34), fill=INK)
            if sat_note and sat_note(a, b):
                d.text((rx, 310), sat_note(a, b), font=F_TXT, fill=WARN)
        else:
            d.text((rx, 210), "Cada pixel recebe a mesma conta:", font=F_BOLD, fill=ACCENT)
        wrap_lines(d, explain, rx, 380, gap=36)
        if phase == "done":
            a, b = full_fn(gray), None
            small = lambda g: Image.fromarray(cv.resize(g, (240, 160), interpolation=cv.INTER_AREA)).convert("RGB")
            im.paste(small(gray), (rx, 520))
            d.text((rx + 262, 580), "→", font=F_TITLE, fill=INK)
            im.paste(small(a), (rx + 320, 520))
            d.text((rx, 686), "a mesma conta, nos 273.280 pixels da foto inteira", font=font("DejaVuSans.ttf", 18), fill=MUTED)
        return im

    v.add(frame(0, "hold"), 2.5)
    for k in range(n * n):
        v.add(frame(k, "calc"), 0.55 if k < 6 else 0.13)
    v.add(frame(n * n, "done"), 5.0)
    v.close()


# ---------- animação 2: estouro do uint8 ----------
def overflow(gray, P):
    name, v = "estouro", None
    v = Video(OUT / f"{name}.mp4")
    add = 150
    n, S = 8, 40
    A = (P.astype(int) + add) % 256            # numpy: dá a volta
    B = np.clip(P.astype(int) + add, 0, 255)   # cv.add: satura
    code = f"wrong = img + {add}       |       ok = cv.add(img, {add})"

    def frame(k):
        im, d = base("Estouro do uint8: 255 + 1 volta a 0", code)
        for gx, M, lab, col in ((60, A, "img + 150   (NumPy)", WARN), (660, B, "cv.add(img, 150)   (satura)", ACCENT)):
            for i in range(n):
                for j in range(n):
                    idx = i * n + j
                    done = idx < k
                    val = M[i, j] if done else P[i, j]
                    over = (int(P[i, j]) + add) > 255
                    hl = col if (done and over) else None
                    draw_cell(d, gx + j * S, 210 + i * S, S, str(int(val)), gray_rgb(val), hl, fsize=15)
            d.text((gx, 210 + n * S + 10), lab, font=F_BOLD, fill=col)
        if 0 <= k < n * n:
            i, j = divmod(k, n)
            a = int(P[i, j])
            d.text((60, 595), f"pixel [{i},{j}]:  {a} + {add} = {a + add}", font=font("DejaVuSansMono-Bold.ttf", 28), fill=INK)
            if a + add > 255:
                d.text((60, 640), f"NumPy dá a volta: {a + add} - 256 = {(a + add) % 256}", font=F_TXT, fill=WARN)
                d.text((660, 640), "cv.add trava no máximo: 255", font=F_TXT, fill=ACCENT)
        return im

    v.add(frame(-1), 2.0)
    for k in range(n * n):
        v.add(frame(k), 0.4 if k < 8 else 0.12)
    v.add(frame(n * n), 4.0)
    v.close()


# ---------- animação 3: mover células (virar, desvirar, rotacionar) ----------
def movement(name, title, code, P, mapping, explain, full_fn, gray, second=None):
    """mapping(i, j) -> (i2, j2): onde o pixel [i, j] vai parar. second: lista de passos extras (desvirar)."""
    v = Video(OUT / f"{name}.mp4")
    n, S, X0, Y0 = 8, 60, 60, 192
    steps = [(title, code, mapping, explain)] + (second or [])

    def frame(cur, t, step_idx, final=False):
        ttl, cd, mp, ex = steps[step_idx]
        im, d = base(ttl, cd)
        for i in range(n):
            for j in range(n):
                i2, j2 = mp(i, j)
                tt = ease(t)
                x = X0 + (j + (j2 - j) * tt) * S
                y = Y0 + (i + (i2 - i) * tt) * S
                draw_cell(d, x, y, S, str(int(cur[i, j])), gray_rgb(cur[i, j]), ACCENT if 0 < t < 1 else None)
        d.text((X0, Y0 + n * S + 14), "os números só trocam de lugar, nenhum valor muda", font=F_TXT, fill=MUTED)
        wrap_lines(d, ex, 640, 220, gap=40)
        if final:
            small = lambda g: Image.fromarray(cv.resize(g, (240, 160), interpolation=cv.INTER_AREA)).convert("RGB")
            im.paste(small(gray), (640, 520))
            d.text((902, 580), "→", font=F_TITLE, fill=INK)
            im.paste(small(full_fn(gray)), (960, 520))
        return im

    cur = P.copy()
    for si, (ttl, cd, mp, ex) in enumerate(steps):
        v.add(frame(cur, 0, si), 2.0)
        for f in range(int(2.2 * FPS)):
            v.add(frame(cur, (f + 1) / (2.2 * FPS), si))
        new = np.zeros_like(cur)
        for i in range(n):
            for j in range(n):
                i2, j2 = mp(i, j)
                new[i2, j2] = cur[i, j]
        cur = new
        v.add(frame(cur, 0, si, final=(si == len(steps) - 1)), 3.0 if si == len(steps) - 1 else 1.5)
    v.close()


# ---------- animação 4: convolução (kernel deslizando) ----------
def convolution(gray, P):
    v = Video(OUT / "convolucao.mp4")
    A = P[:6, :6].astype(np.float32)
    K = np.ones((3, 3), np.float32) / 9
    out = np.zeros((4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            out[i, j] = (A[i:i + 3, j:j + 3] * K).sum()
    S, X0, Y0 = 62, 60, 215
    code = "blur = cv.blur(gray, (3, 3))      # média da vizinhança 3×3"

    def frame(k):
        im, d = base("Blur: um kernel deslizando pela matriz", code)
        for i in range(6):
            for j in range(6):
                inwin = k >= 0 and k < 16 and (k // 4) <= i < (k // 4) + 3 and (k % 4) <= j < (k % 4) + 3
                draw_cell(d, X0 + j * S, Y0 + i * S, S, str(int(A[i, j])), gray_rgb(A[i, j]), ACCENT if inwin else None, fsize=22)
        d.text((X0, Y0 + 6 * S + 10), "entrada 6×6", font=F_TXT, fill=MUTED)
        ox = 520
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < max(k, 0) or k >= 16:
                    draw_cell(d, ox + j * S, Y0 + i * S, S, str(int(round(out[i, j]))), gray_rgb(out[i, j]), WARN if idx == k - 1 else None, fsize=22)
                else:
                    d.rectangle((ox + j * S, Y0 + i * S, ox + j * S + S - 2, Y0 + i * S + S - 2), outline=(190, 196, 205), width=2)
        d.text((ox, Y0 + 4 * S + 10), "saída 4×4", font=F_TXT, fill=MUTED)
        if 0 <= k < 16:
            i, j = divmod(k, 4)
            w = A[i:i + 3, j:j + 3].astype(int)
            s = int(w.sum())
            d.text((800, 215), "soma dos 9 vizinhos ÷ 9", font=F_BOLD, fill=ACCENT)
            for r in range(3):
                d.text((800, 265 + r * 40), "  ".join(f"{x:3d}" for x in w[r]), font=font("DejaVuSansMono-Bold.ttf", 28), fill=INK)
            d.text((800, 400), f"({s}) ÷ 9 = {out[i, j]:.0f}", font=font("DejaVuSansMono-Bold.ttf", 30), fill=INK)
        else:
            wrap_lines(d, ["O kernel é uma matriz pequena", "que desliza e faz uma conta", "em cada vizinhança.", "", "A CNN faz o mesmo, mas", "APRENDE os números do kernel."], 800, 215, gap=38)
        return im

    v.add(frame(-1), 2.0)
    for k in range(16):
        v.add(frame(k), 0.9 if k < 4 else 0.45)
    v.add(frame(16), 4.0)
    v.close()


# ---------- animação 5: BGR -> RGB (3 canais) ----------
def channels(img):
    v = Video(OUT / "bgr_rgb.mp4")
    pat = img[200:204, 320:324].copy()  # 4x4 pixels coloridos
    n, S, X0, Y0 = 4, 120, 60, 200
    code = "rgb = img[:, :, ::-1]      # inverte a ordem dos 3 números de cada pixel"

    def frame(t, hl_t=None):
        im, d = base("Trocar BGR ↔ RGB: cada pixel tem 3 números", code)
        tt = ease(t)
        for i in range(n):
            for j in range(n):
                b, g, r = [int(x) for x in pat[i, j]]
                fill = (r, g, b)  # como o olho vê o pixel ORIGINAL (RGB para o PIL)
                if tt > 0.5:
                    fill = (b, g, r)  # depois de trocar, o pixel vira outra cor
                x, y = X0 + j * S, Y0 + i * S
                d.rectangle((x, y, x + S - 3, y + S - 3), fill=fill)
                trio = [b, g, r] if tt < 0.5 else [r, g, b]
                labels = ["B", "G", "R"] if tt < 0.5 else ["R", "G", "B"]
                for q in range(3):
                    d.text((x + 8, y + 8 + q * 32), f"{labels[q]}={trio[q]}", font=font("DejaVuSansMono-Bold.ttf", 19), fill=(255, 255, 255) if lum(fill) < 140 else INK)
        d.text((X0, Y0 + n * S + 10), "4×4 pixels coloridos: cada um guarda [B, G, R]", font=F_TXT, fill=MUTED)
        wrap_lines(d, ["Antes: ordem B, G, R", "(OpenCV)", "", "Depois: ordem R, G, B", "(matplotlib, PIL, modelos)", "", "Os números continuam os mesmos;", "só mudou a ORDEM dentro de cada pixel."],
                   700, 215, gap=38)
        return im

    v.add(frame(0), 3.0)
    for f in range(int(1.2 * FPS)):
        v.add(frame((f + 1) / (1.2 * FPS)), 0)
    v.add(frame(1), 5.0)
    v.close()


# ---------- definição dos vídeos ----------
def run(only=None):
    img, gray, P = get_data()
    jobs = {}
    jobs["brilho"] = lambda: elementwise(
        "brilho", "Aumentar o brilho: somar um número", "bright = cv.add(gray, 80)", P,
        lambda M: np.clip(M.astype(int) + 80, 0, 255), lambda a, b: f"{a} + 80 = {b}",
        ["Brilho = somar o mesmo número", "em TODOS os pixels.", "", "Mais alto = mais perto do branco."],
        lambda g: cv.add(g, 80), gray)
    jobs["escurecer"] = lambda: elementwise(
        "escurecer", "Diminuir o brilho: subtrair um número", "dark = cv.subtract(gray, 40)", P,
        lambda M: np.clip(M.astype(int) - 40, 0, 255), lambda a, b: f"{a} - 40 = {b}" if a >= 40 else f"{a} - 40 = {a - 40}  → 0",
        ["Escurecer = subtrair.", "", "cv.subtract não deixa passar de 0:", "valor negativo vira 0 (preto)."],
        lambda g: cv.subtract(g, 40), gray, sat_note=lambda a, b: "negativo vira 0 (satura)" if a < 40 else None)
    jobs["contraste"] = lambda: elementwise(
        "contraste", "Contraste: multiplicar", "contrast = cv.convertScaleAbs(gray, alpha=1.5, beta=0)", P,
        lambda M: np.clip((M * 1.5).round(), 0, 255).astype(int), lambda a, b: f"{a} × 1.5 = {b}",
        ["Contraste = multiplicar.", "", "Claros ficam mais claros,", "escuros mudam pouco: a", "diferença entre eles aumenta."],
        lambda g: cv.convertScaleAbs(g, alpha=1.5, beta=0), gray)
    jobs["negativo"] = lambda: elementwise(
        "negativo", "Inverter as cores (negativo): 255 − valor", "inverted = 255 - gray", P,
        lambda M: 255 - M.astype(int), lambda a, b: f"255 - {a} = {b}",
        ["Negativo = 255 menos o pixel.", "", "Preto (0) vira branco (255),", "branco vira preto."],
        lambda g: 255 - g, gray)
    jobs["normalizar"] = lambda: None  # (definido abaixo, valores decimais)
    jobs["threshold"] = lambda: elementwise(
        "threshold", "Threshold: separar em preto e branco", "binary = (gray > 100) * 255", P,
        lambda M: np.where(M > 100, 255, 0), lambda a, b: f"{a} > 100 ?   →   {'sim → 255' if a > 100 else 'não → 0'}",
        ["Threshold = uma pergunta por pixel:", "“é maior que 100?”", "", "Sim → branco (255)", "Não → preto (0)"],
        lambda g: ((g > 100) * 255).astype("uint8"), gray)
    jobs["mascara"] = lambda: elementwise(
        "mascara", "Máscara: deixar só uma região aparente", "only = cv.bitwise_and(gray, gray, mask=mask)", P,
        lambda M: np.where(np.arange(64).reshape(8, 8) % 8 >= 4, M, 0).astype(int) * 1,
        lambda a, b: f"{a} AND 255 = {a}" if b else f"{a} AND 0 = 0",
        ["Máscara = matriz de 0 e 255.", "", "255 (branco): o pixel passa", "0 (preto): o pixel some", "", "(aqui: só as 4 colunas da direita)"],
        lambda g: np.where(np.arange(g.shape[1])[None, :] >= g.shape[1] // 2, g, 0).astype("uint8"), gray)
    jobs["estouro"] = lambda: overflow(gray, P)
    jobs["virar"] = lambda: movement(
        "virar", "Virar a imagem (espelho): inverter as colunas", "h = gray[:, ::-1]", P, lambda i, j: (i, 7 - j),
        ["h[i, j] = gray[i, 7 - j]", "", "A 1ª coluna vai para a última,", "a 2ª para a penúltima..."],
        lambda g: g[:, ::-1], gray)
    jobs["desvirar"] = lambda: movement(
        "desvirar", "Virar e desvirar: a mesma operação duas vezes", "h = gray[:, ::-1]", P, lambda i, j: (i, 7 - j),
        ["1) Virar: inverte as colunas."],
        lambda g: g, gray,
        second=[("Desvirar = virar de novo", "back = h[:, ::-1]", lambda i, j: (i, 7 - j),
                 ["2) Desvirar: inverter outra vez.", "", "np.array_equal(back, gray) → True", "Voltou igual, pixel a pixel."])])
    jobs["vertical"] = lambda: movement(
        "vertical", "Virar de cabeça para baixo: inverter as linhas", "v = gray[::-1]", P, lambda i, j: (7 - i, j),
        ["v[i, j] = gray[7 - i, j]", "", "A 1ª linha vai para a última,", "a 2ª para a penúltima..."],
        lambda g: g[::-1], gray)
    jobs["rotacionar"] = lambda: movement(
        "rotacionar", "Rotacionar 90°: o pixel [i, j] vai para [j, 7 − i]", "r90 = cv.rotate(gray, cv.ROTATE_90_CLOCKWISE)", P,
        lambda i, j: (j, 7 - i),
        ["Cada pixel muda de lugar", "seguindo uma regra fixa.", "", "A matriz continua 8×8 aqui;", "numa foto retangular, altura", "e largura trocam."],
        lambda g: cv.rotate(g, cv.ROTATE_90_CLOCKWISE), gray)
    jobs["convolucao"] = lambda: convolution(gray, P)
    jobs["bgr_rgb"] = lambda: channels(img)

    def normalizar():
        v = Video(OUT / "normalizar.mp4")
        n, S, X0, Y0 = 8, 60, 60, 192
        Q = P.astype(np.float32) / 255

        def frame(k, done=False):
            im, d = base("Normalizar: dividir por 255", 'x = gray.astype("float32") / 255.0')
            for i in range(n):
                for j in range(n):
                    idx = i * n + j
                    after = idx < k
                    txt = f"{Q[i, j]:.2f}" if after else str(int(P[i, j]))
                    draw_cell(d, X0 + j * S, Y0 + i * S, S, txt, gray_rgb(P[i, j]), ACCENT if idx == k else None, fsize=21)
            d.text((X0, Y0 + n * S + 14), "valores de 0–255 viram 0.0–1.0", font=F_TXT, fill=MUTED)
            if 0 <= k < 64:
                i, j = divmod(k, n)
                d.text((640, 215), f"pixel [{i}, {j}]", font=F_BOLD, fill=ACCENT)
                d.text((640, 260), f"{int(P[i, j])} ÷ 255 = {Q[i, j]:.2f}", font=font("DejaVuSansMono-Bold.ttf", 34), fill=INK)
            wrap_lines(d, ["Por quê? A rede treina melhor", "com números pequenos e na", "mesma escala.", "", "A imagem é a mesma:", "só mudou a régua."], 640, 380, gap=38)
            return im
        v.add(frame(-1), 2.5)
        for k in range(64):
            v.add(frame(k), 0.5 if k < 6 else 0.12)
        v.add(frame(64), 4.5)
        v.close()
    jobs["normalizar"] = normalizar

    names = [only] if only else list(jobs)
    for nme in names:
        print("gerando", nme, "...", flush=True)
        jobs[nme]()
    print("ok →", OUT)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
