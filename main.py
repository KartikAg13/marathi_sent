import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from keras import Sequential
from keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    SpatialDropout1D,
)

from keras.utils import pad_sequences

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras_preprocessing.text import Tokenizer

from sklearn.metrics import classification_report, confusion_matrix

import pickle

np.random.seed(seed=42)
tf.random.set_seed(seed=42)

LOAD_PATH: str = "dataset/preprocess/"
MODEL_PATH: str = "model/bilstm_marathi.keras"


class BiLSTM:
    def __init__(
        self, max_words: int, max_len: int, embedding_dim: int, lstm_units: int
    ) -> None:
        self.max_words: int = max_words
        self.max_len: int = max_len
        self.embedding_dim: int = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.lstm_units: int = lstm_units
        self.model: Sequential | None = None
        self.history = None

        self.labels: dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self.reverse_labels: dict[int, int] = {0: -1, 1: 0, 2: 1}

    def encode(self, text: pd.Series) -> np.ndarray:
        """Convert text to padded sequences"""
        seq: list = self.tokenizer.texts_to_sequences(texts=text)
        return pad_sequences(
            sequences=seq, maxlen=self.max_len, padding="post", truncating="post"
        )

    def tokenize(
        self, train_text: pd.Series, test_text: pd.Series, val_text: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit tokenizer on training data and encode all datasets"""
        self.tokenizer.fit_on_texts(texts=train_text)
        print(f"Vocabulary Size: {len(self.tokenizer.word_index)}")
        return (self.encode(train_text), self.encode(test_text), self.encode(val_text))

    def maping(self, labels: pd.Series) -> np.ndarray:
        """Map labels from {-1, 0, 1} to {0, 1, 2}"""
        return np.array([self.labels[label] for label in labels])

    def mapLabels(
        self, train_labels: pd.Series, test_labels: pd.Series, val_labels: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map all label sets"""
        return (
            self.maping(train_labels),
            self.maping(test_labels),
            self.maping(val_labels),
        )

    def build(self, dropout_rate: float) -> Sequential:
        """Build the BiLSTM model architecture"""
        vocab_size = min(self.max_words, len(self.tokenizer.word_index)) + 1
        model: Sequential = Sequential(
            [
                Embedding(
                    input_dim=vocab_size,
                    output_dim=self.embedding_dim,
                    input_length=self.max_len,
                    mask_zero=True,
                ),
                SpatialDropout1D(dropout_rate),
                Bidirectional(LSTM(units=self.lstm_units)),
                Dropout(dropout_rate),
                Dense(units=64, activation="relu"),
                Dropout(dropout_rate),
                Dense(units=3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        self.model = model
        model.build(input_shape=(None, self.max_len))
        model.summary()
        return model

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        """Train the model with callbacks"""
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-7,
            verbose=1,
        )

        checkpoint = ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        )

        if self.model is not None:
            self.history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose="auto",
            )
        return self.history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model on test set"""
        if self.model is not None:
            loss, accuracy = self.model.evaluate(x=x_test, y=y_test, verbose="auto")
            y_pred_prob = self.model.predict(x=x_test, verbose="auto")
        else:
            raise ValueError("Model not built. Call build() first.")

        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_pred: np.ndarray = np.argmax(a=y_pred_prob, axis=1)

        y_test_original = np.array([self.reverse_labels[label] for label in y_test])
        y_pred_original = np.array([self.reverse_labels[label] for label in y_pred])

        print("\nClassification Report:")
        cr = classification_report(
            y_true=y_test_original,
            y_pred=y_pred_original,
            target_names=["Negative(-1)", "Neutral(0)", "Positive(1)"],
        )
        print(cr)

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true=y_test_original, y_pred=y_pred_original)
        print(cm)

        return (loss, accuracy, cr, cm, y_pred_original)

    def plotHistory(self) -> None:
        """Plot training history"""
        if self.history is None:
            print("Training not completed")
            return

        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(
            self.history.history["accuracy"], label="Train Accuracy", linewidth=2
        )
        axes[0].plot(
            self.history.history["val_accuracy"], label="Val Accuracy", linewidth=2
        )
        axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Accuracy", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history.history["loss"], label="Train Loss", linewidth=2)
        axes[1].plot(self.history.history["val_loss"], label="Val Loss", linewidth=2)
        axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plotCM(self, cm) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"],
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()


def getXY(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract features and labels from dataframe"""
    return data["text"], data["label"]


def main() -> None:
    # Load data
    train_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}train.csv", encoding="utf-8")
    test_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}test.csv", encoding="utf-8")
    val_df: pd.DataFrame = pd.read_csv(f"{LOAD_PATH}val.csv", encoding="utf-8")

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Extract features and labels
    x_train, y_train = getXY(train_df)
    x_test, y_test = getXY(test_df)
    x_val, y_val = getXY(val_df)

    # Initialize model
    model: BiLSTM = BiLSTM(
        max_words=15000,
        max_len=150,
        embedding_dim=256,
        lstm_units=256,
    )

    # Tokenize texts
    print("\nTokenizing texts...")
    x_train, x_test, x_val = model.tokenize(x_train, x_test, x_val)

    # Map labels
    print("Mapping labels...")
    y_train, y_test, y_val = model.mapLabels(y_train, y_test, y_val)

    # Build model
    print("\nBuilding model...")
    _ = model.build(dropout_rate=0.25)

    # Train model
    print("\nTraining model...")
    batch_size: int = 64
    _ = model.train(x_train, y_train, x_val, y_val, epochs=5, batch_size=batch_size)

    # Plot training history
    model.plotHistory()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    (loss, accuracy, cr, cm, y_pred) = model.evaluate(x_test, y_test)

    # Plot confusion matrix
    model.plotCM(cm)

    print("\nSaving tokenizer...")
    with open("model/tokenizer.pkl", "wb") as f:
        pickle.dump(model.tokenizer, f)
    print("Tokenizer saved to model/tokenizer.pkl")


if __name__ == "__main__":
    main()
