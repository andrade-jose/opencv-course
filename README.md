# OpenCV com Python em 4 horas (versão atualizada)

Código e anotações do curso [**Python and OpenCV**](https://youtu.be/oXlwWbU8l2o) do Jason Dsouza (freeCodeCamp.org), **atualizados para o OpenCV e o ecossistema Python atuais**.

Este repositório é uma atualização de [jasmcaus/opencv-course](https://github.com/jasmcaus/opencv-course) (licença MIT, mantida). O código original é de 2020 e parou de funcionar em vários pontos; aqui tudo foi revisado e testado.

## O que mudou em relação ao original

| Área | Antes | Agora |
|---|---|---|
| Capstone (Simpsons) | usava `caer` e `canaro` (`canaro` não existe mais no PyPI) | **NumPy + OpenCV + scikit-learn + Keras 3**, sem `caer`/`canaro` |
| Keras | `optimizers.legacy.SGD` (gera `ImportError` no Keras 3) e `decay` (ignorado) | `keras.optimizers.SGD` + `LearningRateScheduler` |
| Capstone: perda | `binary_crossentropy` com 10 classes | `categorical_crossentropy` |
| Capstone: predição | imagem de teste **não** era normalizada (treino em [0,1], teste em [0,255]) | mesmo pré-processamento no treino e no teste |
| Capstone: augmentation | `ImageDataGenerator` via `canaro` | camadas `RandomFlip`/`RandomRotation`/`RandomZoom` dentro do modelo |
| Reconhecimento facial | caminhos com `\` do Windows (só funcionava lá) e lista de nomes duplicada; modelo de 12 MB (YAML sem compressão) | caminhos com `pathlib` (Windows, Linux e Mac), lista de pessoas lida das pastas, modelo retreinado e comprimido (`.yml.gz`, 0,8 MB) |
| `dilate`/`erode` | kernel passado como tupla `(7,7)` (roda, mas o OpenCV o interpreta como um kernel 2×1, **não** 7×7) | `np.ones((7, 7), np.uint8)` |
| `gradients.py` | `bitwise_or` em `float64` (roda, mas dá uma imagem sem sentido) | `convertScaleAbs` antes do `bitwise_or` |
| `rescale_resize.py` | **crash** no fim do vídeo (`frame` vazio) | para quando o vídeo acaba |
| Caminhos | `'../Resources/...'` (só funcionava rodando de dentro da pasta) | funcionam de qualquer diretório |
| Dependência | `opencv-python` + `caer` | `opencv-contrib-python` (o módulo `cv.face` só existe no *contrib*) |

Também foram renomeadas as pastas e arquivos (sem espaços, `#` ou maiúsculas) e corrigidos nomes: `Ben Afflek` → `ben_affleck`, `Jerry Seinfield` → `jerry_seinfeld`.

> **Nota sobre o OpenCV 5:** o pacote pip do OpenCV 5.0 **não traz mais os arquivos `haarcascade_*.xml`** em `cv2.data`. Por isso o `haar_face.xml` continua incluído em `03_faces/` e é carregado direto dele.

## Estrutura

```
00_aula/        Gabaritos da aula "a imagem é uma matriz": um script por slide (rode a partir de qualquer pasta)
01_basics/      Seção 1: ler imagens/vídeos, desenhar, transformações, contornos, threshold
02_advanced/    Seção 2: espaços de cor, blur, bitwise, máscaras, histogramas, gradientes
03_faces/       Seção 3: detecção (Haar Cascade) e reconhecimento facial (LBPH)
04_capstone/    Seção 4: CNN para classificar personagens de Os Simpsons (Keras)
resources/      Fotos, vídeos e rostos usados nos exemplos
tests/          smoke_test.py: roda todos os scripts sem abrir janelas
```

## Como rodar

Requer **Python 3.10+**.

```bash
git clone <URL-DESTE-REPOSITORIO>
cd <pasta-do-repositorio>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # Seções 1 a 3
```

Cada script pode ser executado de qualquer diretório:

```bash
python 01_basics/read.py
python 02_advanced/blurring.py
python 03_faces/face_detect.py
```

Nas janelas do OpenCV, qualquer tecla avança (`waitKey(0)`); nos vídeos, pressione `d` para sair.

> **Importante:** instale **apenas um** dos pacotes `opencv-python`, `opencv-contrib-python` ou `opencv-python-headless`. Eles se sobrescrevem. Neste repositório use o `opencv-contrib-python`. Se já tiver outro instalado: `pip uninstall opencv-python opencv-python-headless` e depois `pip install opencv-contrib-python`.
>
> Em servidores sem interface gráfica (sem monitor), `cv.imshow` não funciona; use `python tests/smoke_test.py` para validar os scripts.

### Seção 3: reconhecimento facial

O modelo treinado (`face_trained.yml.gz`) já está no repositório. Para treinar de novo ou usar suas próprias fotos (uma subpasta por pessoa em `resources/faces/train/`):

```bash
python 03_faces/faces_train.py
python 03_faces/face_recognition.py                  # foto de exemplo
python 03_faces/face_recognition.py minha_foto.jpg   # sua foto
```

Com as 5 pessoas e as poucas fotos do dataset, o LBPH acerta cerca de 14 de 25 imagens de validação (~56%). É uma boa deixa para discutir os limites de métodos clássicos frente a redes neurais.

### Seção 4: capstone (Simpsons)

```bash
pip install -r requirements-capstone.txt
```

1. Baixe o [dataset do Kaggle](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset) e extraia.
2. Aponte para a pasta que contém `simpsons_dataset/` e `kaggle_simpson_testset/`:
   ```bash
   export SIMPSONS_DIR=/caminho/para/the-simpsons-characters-dataset   # Windows: set SIMPSONS_DIR=...
   python 04_capstone/simpsons.py
   ```
3. Ou abra `04_capstone/simpsons.ipynb` no Kaggle/Colab (o caminho padrão já é o do Kaggle: `../input/the-simpsons-characters-dataset`).

O treino gera `simpsons.keras` (ignorado pelo git por ter ~140 MB).

## Versões testadas

Python 3.13, OpenCV 5.0.0 e 4.13.0, NumPy 2.5, Matplotlib 3.11, TensorFlow 2.21 / Keras 3.15, scikit-learn 1.9.
As seções 1 a 3 foram executadas de ponta a ponta nas duas versões do OpenCV (`python tests/smoke_test.py`). O capstone foi validado de ponta a ponta com um dataset sintético; **não** foi treinado com o dataset real do Kaggle.

## Conteúdo do curso (com marcações de tempo do vídeo original)

### 1. Instalação
Instalação do OpenCV (`pip install opencv-contrib-python`). O pacote `caer`, usado no vídeo, **não é mais necessário**.

### 2. Conceitos básicos
- Lendo imagens e vídeos ([0:04:12](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=252s))
- Redimensionando imagens e frames de vídeo ([0:12:57](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=777s))
- Desenhando formas e escrevendo texto ([0:20:21](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=1221s))
- 5 funções essenciais do OpenCV ([0:31:55](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=1915s))
- Transformações de imagem ([0:44:13](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=2653s))
- Detecção de contornos ([0:57:06](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=3426s))

### 3. Conceitos avançados
- Espaços de cor: RGB, BGR, Grayscale, HSV e L\*a\*b ([1:12:53](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=4373s))
- Separando e juntando canais de cor ([1:23:10](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=4990s))
- Blur ([1:31:03](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=5463s))
- Operações bitwise ([1:44:27](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=6267s))
- Máscaras ([1:53:06](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=6786s))
- Histogramas ([2:01:43](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=7303s))
- Thresholding/binarização ([2:15:22](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=8122s))
- Detecção avançada de bordas ([2:26:27](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=8787s))

### 4. Detecção e reconhecimento facial
- Detecção de rostos com Haar Cascades ([2:35:25](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=9325s))
- Reconhecimento facial com o `LBPHFaceRecognizer` do OpenCV ([2:49:05](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=10145s))

### 5. Capstone: visão computacional com deep learning
- Uma CNN para classificar personagens da série Os Simpsons ([3:11:57](https://www.youtube.com/watch?v=x3c8w2ruhjs&t=11517s))

## Créditos

- Curso e código original: [Jason Dsouza](https://github.com/jasmcaus), repositório [jasmcaus/opencv-course](https://github.com/jasmcaus/opencv-course), licença MIT.
- As imagens e vídeos em `resources/photos` e `resources/videos` vêm do [Unsplash](http://unsplash.com) e do [Pixabay](http://pixabay.com), salvo indicação em contrário.
- As imagens em `resources/faces` vêm do [5 Celebrity Faces Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) (Kaggle).
