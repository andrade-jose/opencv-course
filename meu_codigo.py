"""Seu arquivo de prática da aula. Cole o código de cada slide na área marcada e rode.

Rode com:  python meu_codigo.py   (ou pelo botão "Run" do editor; funciona de qualquer pasta)
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # garante que "resources/..." sempre é encontrado

import cv2 as cv
import numpy as np

img = cv.imread("resources/photos/park.jpg")
assert img is not None, "Foto não encontrada: o meu_codigo.py precisa ficar na pasta principal do projeto"
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ======== COLE O CÓDIGO DO SLIDE AQUI (debaixo desta linha) ========



# ===================================================================
cv.waitKey(0)  # mantém as janelas abertas; clique numa janela e aperte qualquer tecla para fechar
cv.destroyAllWindows()
