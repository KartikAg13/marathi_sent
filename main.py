import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf

# from keras import Sequential, Model
# from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Input
from keras.utils import pad_sequences

# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.optimizers import Adam
from keras_preprocessing.text import Tokenizer

# from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(seed=42)
tf.random.set_seed(seed=42)

LOAD_PATH: str = "dataset/preprocess/"


class BiLSTM:
    def __init__(self, max_words: int, max_len: int, embedding_dim: int) -> None:
        self.max_words: int = max_words
        self.max_len: int = max_len
        self.embedding_dim: int = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
        self.history = None

        self.labels: dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.reverse_labels: dict[int, int] = {0: -1, 1: 0, 2: 1}

    def encode(self, text: pd.Series) -> np.ndarray:
        seq: list = self.tokenizer.texts_to_sequences(texts=text)
        return pad_sequences(
            sequences=seq, maxlen=self.max_len, padding="post", truncating="post"
        )

    def tokenize(
        self, train_text: pd.Series, test_text: pd.Series, val_text: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.tokenizer.fit_on_texts(texts=train_text)
        print(f"Vocabulary Size: {len(self.tokenizer.word_index)}")
        return (self.encode(train_text), self.encode(test_text), self.encode(val_text))

    def maping(self, labels: pd.Series) -> np.ndarray:
        return np.array([self.labels[label] for label in labels])

    def mapLabels(
        self, train_labels: pd.Series, test_labels: pd.Series, val_labels: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.maping(train_labels),
            self.maping(test_labels),
            self.maping(val_labels),
        )


def getXY(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return data["text"], data["label"]


def main() -> None:
    train_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}train.csv", encoding="utf-8")
    test_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}test.csv", encoding="utf-8")
    val_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}val.csv", encoding="utf-8")

    x_train, y_train = getXY(train_df)
    x_test, y_test = getXY(test_df)
    x_val, y_val = getXY(val_df)

    model: BiLSTM = BiLSTM(max_words=15000, max_len=150, embedding_dim=128)
    x_train, x_test, x_val = model.tokenize(x_train, x_test, x_val)
    y_train, y_test, y_val = model.mapLabels(y_train, y_test, y_val)


if __name__ == "__main__":
    main()
