# evaluate.py
"""
Evaluate saved BiLSTM model on test set.
Outputs:
 - RMSE
 - Accuracy
 - Precision (macro)
 - Recall (macro)
 - F1 (macro)
 - Classification report (per-class)
 - Confusion matrix (printed + plotted)
Saves metrics to model/metrics.json and confusion matrix image.
Author: Kartik (adapted)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
)


DEFAULT_MODEL = "model/bilstm_marathi.keras"
DEFAULT_TOKENIZER_PATHS = [
    "model/tokenizer.pkl",
]
DEFAULT_TEST_CSV = "dataset/preprocess/test.csv"


def load_tokenizer(paths):
    """Try several candidate paths and return loaded tokenizer or raise."""
    for p in paths:
        p = Path(p)
        if p.exists():
            with open(p, "rb") as f:
                tokenizer = pickle.load(f)
            print(f"✅ Loaded tokenizer from: {p}")
            return tokenizer
    raise FileNotFoundError(f"No tokenizer found in {paths}")


def prepare_sequences(tokenizer, texts, max_len):
    """Convert raw texts to padded sequences using the tokenizer."""
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded


def canonical_label_mapping(y):
    """
    Given label array y from CSV, return:
     - y_original: labels as they are (numpy array)
     - encoded_y: labels encoded to [0,1,2] according to mapping used in training:
         mapping: {-1->0, 0->1, 1->2}
     - reverse_map: dict to convert encoded -> original
    Detects if input labels are already encoded (0,1,2).
    """
    unique = set(np.unique(y).tolist())
    # if labels already in {0,1,2}
    if unique.issubset({0, 1, 2}):
        # assume already encoded
        encoded = y.astype(int)
        reverse_map = {0: -1, 1: 0, 2: 1}  # keep reverse map consistent with training
        # attempt to recover original labels by applying reverse_map (if the user had stored original)
        original = np.array([reverse_map[int(lb)] for lb in encoded])
        return original, encoded, reverse_map

    # If labels in {-1,0,1} or similar
    if unique.issubset({-1, 0, 1}):
        mapping = {-1: 0, 0: 1, 1: 2}
        reverse_map = {0: -1, 1: 0, 2: 1}
        encoded = np.array([mapping[int(lb)] for lb in y])
        original = np.array([int(lb) for lb in y])
        return original, encoded, reverse_map

    # Fallback: try to infer monotonic mapping (smallest->0,...)
    sorted_uniques = sorted(list(unique))
    mapping = {v: i for i, v in enumerate(sorted_uniques)}
    reverse_map = {i: v for v, i in mapping.items()}
    encoded = np.array([mapping[int(lb)] for lb in y])
    original = np.array([int(lb) for lb in y])
    print(
        "⚠️ Warning: labels were not standard {-1,0,1} or {0,1,2}. "
        f"Applied auto-mapping: {mapping}"
    )
    return original, encoded, reverse_map


def evaluate_predictions(y_true_original, y_pred_original, labels_display=None):
    """Compute metrics and return dictionary."""
    # numeric arrays
    y_true = np.array(y_true_original)
    y_pred = np.array(y_pred_original)

    # RMSE computed on numeric original labels (e.g., -1,0,1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # classification report (string and dict)
    cls_report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    cls_report_str = classification_report(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "rmse": rmse,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classification_report": cls_report_dict,
    }

    return metrics, cls_report_str, cm


def plot_and_save_cm(cm, labels, out_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to {out_path}")


def extract_model_hyperparams(model):
    """Extract key hyperparameters from the model."""
    params = {}

    for layer in model.layers:
        cfg = layer.get_config()
        name = layer.__class__.__name__

        if name == "Embedding":
            params["embedding"] = {
                "input_dim": cfg.get("input_dim"),
                "output_dim": cfg.get("output_dim"),
                "input_length": cfg.get("input_length"),
                "mask_zero": cfg.get("mask_zero"),
            }

        elif name == "Bidirectional":
            # ✅ Handle both older and newer TensorFlow versions
            lstm_layer = getattr(layer, "layer", None)
            if lstm_layer is None and hasattr(layer, "_layers"):
                lstm_layer = layer._layers[0]  # safely access first internal LSTM

            if lstm_layer:
                lstm_cfg = lstm_layer.get_config()
                params["bilstm"] = {
                    "units": lstm_cfg.get("units"),
                    "recurrent_dropout": lstm_cfg.get("recurrent_dropout"),
                    "return_sequences": lstm_cfg.get("return_sequences"),
                    "go_backwards": lstm_cfg.get("go_backwards"),
                    "dropout": lstm_cfg.get("dropout"),
                }

        elif name == "LSTM":  # if not bidirectional
            params["lstm"] = {
                "units": cfg.get("units"),
                "recurrent_dropout": cfg.get("recurrent_dropout"),
                "return_sequences": cfg.get("return_sequences"),
                "dropout": cfg.get("dropout"),
            }

        elif name == "Dropout":
            params.setdefault("dropout_layers", []).append(cfg.get("rate"))

        elif name == "Dense":
            params["dense"] = {
                "units": cfg.get("units"),
                "activation": cfg.get("activation"),
            }

    # ✅ Extract optimizer, loss, metrics safely
    try:
        params["optimizer"] = {
            "type": model.optimizer.__class__.__name__,
            "learning_rate": float(model.optimizer.learning_rate.numpy()),
        }
    except Exception:
        params["optimizer"] = {"type": "Unknown", "learning_rate": None}

    try:
        params["loss"] = getattr(model.loss, "__name__", str(model.loss))
    except Exception:
        params["loss"] = "Unknown"

    try:
        params["metrics"] = [m.name for m in model.metrics]
    except Exception:
        params["metrics"] = []

    return params


def main(args):
    model_path = Path(args.model)
    tokenizer_paths = (
        args.tokenizer if isinstance(args.tokenizer, list) else [args.tokenizer]
    )
    test_csv = Path(args.test)
    max_len = int(args.max_len)

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(str(model_path))
    print(f"✅ Loaded model from: {model_path}")

    hyperparams = extract_model_hyperparams(model)
    print("\n=== Model Hyperparameters ===")
    print(json.dumps(hyperparams, indent=4))

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_paths)

    # Load test CSV
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    df_test = pd.read_csv(test_csv, encoding="utf-8")
    if "text" not in df_test.columns or "label" not in df_test.columns:
        raise ValueError("Test CSV must contain 'text' and 'label' columns")

    texts = df_test["text"].astype(str).tolist()
    labels_raw = df_test["label"].to_numpy()

    # Prepare sequences
    X_test = prepare_sequences(tokenizer, texts, max_len)

    # Prepare labels: get original numeric labels and encoded (0/1/2)
    y_original, y_encoded, reverse_map = canonical_label_mapping(labels_raw)

    # Predict
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)

    # Convert encoded predictions back to original label space using reverse_map
    y_pred_original = np.array([reverse_map[int(i)] for i in y_pred_encoded])

    # Evaluate
    metrics, cls_report_str, cm = evaluate_predictions(y_original, y_pred_original)

    # Print nicely
    print("\n=== Aggregate metrics ===")
    print(f"RMSE:             {metrics['rmse']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"Recall (macro):   {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):       {metrics['f1_macro']:.4f}")

    print("\n=== Classification Report ===")
    print(cls_report_str)

    print("\n=== Confusion Matrix ===")
    print(cm)

    # Save confusion matrix plot
    # labels for display in CM: unique sorted original labels
    labels_display = sorted(list(set(y_original)))
    plot_and_save_cm(cm, labels=[str(l) for l in labels_display], out_path=args.cm_out)

    # Save metrics to JSON
    out_metrics_path = Path(args.metrics_out)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"✅ Metrics JSON saved to: {out_metrics_path}")

    # Optionally save predictions with true labels to CSV
    if args.save_predictions:
        out_df = pd.DataFrame(
            {
                "text": texts,
                "y_true": y_original,
                "y_pred": y_pred_original,
                "y_pred_encoded": y_pred_encoded.tolist(),
            }
        )
        out_df.to_csv(args.pred_out, index=False, encoding="utf-8")
        print(f"✅ Predictions saved to: {args.pred_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved BiLSTM model.")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Path to saved .keras model"
    )
    parser.add_argument(
        "--tokenizer",
        nargs="+",
        default=DEFAULT_TOKENIZER_PATHS,
        help="Path(s) to tokenizer pickle (first found will be used)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=DEFAULT_TEST_CSV,
        help="Path to test CSV (must have text,label)",
    )
    parser.add_argument(
        "--max_len", type=int, default=125, help="Max sequence length used by the model"
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="model/metrics.json",
        help="Where to save metrics JSON",
    )
    parser.add_argument(
        "--cm_out",
        type=str,
        default="model/confusion_matrix.png",
        help="Where to save confusion matrix image",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-sample predictions to CSV",
    )
    parser.add_argument(
        "--pred_out",
        type=str,
        default="model/predictions.csv",
        help="Where to save predictions CSV",
    )
    args = parser.parse_args()
    main(args)
