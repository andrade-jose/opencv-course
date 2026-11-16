"""Capstone: classificando personagens de Os Simpsons com uma CNN (Keras 3 + OpenCV).

Versão atualizada: não usa mais `caer` nem `canaro`. Tudo é feito com OpenCV, NumPy,
scikit-learn e Keras.

Dataset (Kaggle): https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset
Defina o caminho com a variável de ambiente SIMPSONS_DIR (pasta que contém `simpsons_dataset/`
e `kaggle_simpson_testset/`), ou edite DATASET_DIR abaixo.

Uso:  python simpsons.py
"""
import os
from pathlib import Path

import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split

IMG_SIZE = (80, 80)  # (largura, altura) -- é a ordem esperada pelo cv.resize
CHANNELS = 1  # 1 = escala de cinza
NUM_CLASSES = 10  # usaremos os 10 personagens com mais imagens
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

DATASET_DIR = Path(os.environ.get("SIMPSONS_DIR", "../input/the-simpsons-characters-dataset"))
TRAIN_DIR = DATASET_DIR / "simpsons_dataset"
TEST_DIR = DATASET_DIR / "kaggle_simpson_testset" / "kaggle_simpson_testset"

keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Os 10 personagens com mais imagens (substitui caer.sort_dict)
# ---------------------------------------------------------------------------
if not TRAIN_DIR.is_dir():
    raise SystemExit(
        f"Dataset não encontrado em {TRAIN_DIR}.\n"
        "Baixe do Kaggle e defina SIMPSONS_DIR (veja o topo deste arquivo)."
    )

char_counts = {char.name: len(list(char.iterdir())) for char in TRAIN_DIR.iterdir() if char.is_dir()}
characters = sorted(char_counts, key=char_counts.get, reverse=True)[:NUM_CLASSES]
print("Personagens:", characters)


# ---------------------------------------------------------------------------
# 2. Carregando e pré-processando as imagens (substitui caer.preprocess_from_dir/sep_train)
# ---------------------------------------------------------------------------
def load_dataset():
    images, labels = [], []
    for label, char in enumerate(characters):
        for img_path in sorted((TRAIN_DIR / char).iterdir()):
            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if img is None:  # arquivo corrompido ou que não é imagem
                continue
            images.append(cv.resize(img, IMG_SIZE, interpolation=cv.INTER_AREA))
            labels.append(label)
    # (N, altura, largura, 1) e normalização para [0, 1] (substitui caer.normalize)
    x = np.array(images, dtype="float32").reshape(-1, IMG_SIZE[1], IMG_SIZE[0], CHANNELS) / 255.0
    y = np.array(labels)
    return x, y


features, labels = load_dataset()
print(f"{len(features)} imagens carregadas, formato {features.shape}")

# Visualizando uma amostra (o cv.imshow não funciona bem em notebooks, usamos o matplotlib)
plt.figure(figsize=(4, 4))
plt.imshow(features[0].squeeze(), cmap="gray")
plt.title(characters[labels[0]])
plt.show()

# ---------------------------------------------------------------------------
# 3. Divisão treino/validação (substitui caer.train_val_split)
#    `stratify` mantém a proporção de cada personagem nos dois conjuntos
# ---------------------------------------------------------------------------
x_train, x_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=SEED, stratify=labels
)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

# ---------------------------------------------------------------------------
# 4. Modelo (a augmentation faz parte do modelo e só atua durante o treino;
#    substitui canaro.generators.imageDataGenerator)
# ---------------------------------------------------------------------------
model = keras.Sequential(
    [
        keras.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], CHANNELS)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),  # camada de saída
    ]
)
model.summary()


# ---------------------------------------------------------------------------
# 5. Treinamento
# ---------------------------------------------------------------------------
def lr_schedule(epoch, lr):
    """Reduz a taxa de aprendizado 10x a cada 10 épocas (substitui canaro.lr_schedule)."""
    return 1e-3 * (0.1 ** (epoch // 10))


# Em Keras 3 o argumento `decay` é ignorado (com aviso) e `optimizers.legacy` gera ImportError.
optimizer = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
# Classificação com 10 classes e rótulos one-hot -> categorical_crossentropy
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

training = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule)],
)

model.save(Path(__file__).resolve().parent / "simpsons.keras")

# ---------------------------------------------------------------------------
# 6. Testando com uma imagem do conjunto de teste
# ---------------------------------------------------------------------------
test_path = TEST_DIR / "charles_montgomery_burns_0.jpg"
img = cv.imread(str(test_path))
if img is None:
    raise SystemExit(f"Imagem de teste não encontrada: {test_path}")

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # matplotlib espera RGB
plt.show()


def prepare(image):
    """Mesmo pré-processamento do treino: cinza -> resize -> [0,1] -> (1, h, w, 1)."""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE, interpolation=cv.INTER_AREA)
    image = image.astype("float32") / 255.0
    return image.reshape(1, IMG_SIZE[1], IMG_SIZE[0], CHANNELS)


predictions = model.predict(prepare(img))

# Classe com a maior probabilidade
print(characters[np.argmax(predictions[0])])
