"""Train, evaluate and save the best model.

Run from the project root:  python -m digitlab.train
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from .pipeline import candidate_models, compare_models, load_data


def plot_confusion(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mistakes(x_test, y_true, y_pred, path: Path, limit: int = 12) -> None:
    wrong = np.flatnonzero(y_true != y_pred)[:limit]
    if len(wrong) == 0:
        return
    cols = min(6, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(x_test[i].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"true {y_true[i]}\npred {y_pred[i]}", fontsize=9)
    fig.suptitle("Digits the model got wrong", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()

    models = candidate_models()
    ranking = compare_models(models, x_train, y_train)
    print("5-fold cross-validation on the training set:")
    for r in ranking:
        print(f"  {r.name:<22} {r.mean:.3f} +/- {r.std:.3f}")

    best_name = ranking[0].name
    best = clone(models[best_name]).fit(x_train, y_train)  # refit on all training data
    predictions = best.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    print(f"\nBest model: {best_name}. Held-out test accuracy: {accuracy:.3f}\n")
    print(classification_report(y_test, predictions, digits=3))

    joblib.dump(best, out_dir / "model.joblib")
    plot_confusion(y_test, predictions, out_dir / "confusion_matrix.png")
    plot_mistakes(x_test, y_test, predictions, out_dir / "mistakes.png")

    summary = {
        "best_model": best_name,
        "test_accuracy": round(accuracy, 4),
        "cross_validation": [
            {"model": r.name, "mean_accuracy": round(r.mean, 4), "std": round(r.std, 4)} for r in ranking
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="where to write the model and plots")
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
