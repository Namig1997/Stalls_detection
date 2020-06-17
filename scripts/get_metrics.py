import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score)
import argparse


__folder_current = os.path.join(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_res = os.path.join(__folder, "res")
__folder_data = os.path.join(__folder_res, "data")
filename_train_labels = os.path.join(__folder_data, "train_labels.csv")


def get_binary(y, threshold=0.5):
    return y >= threshold

def get_scores(y_true, y_pred, threshold=0.5):
    y_pred_bin = get_binary(y_pred, threshold=threshold)
    scores = {
        "ratio": np.sum(y_pred_bin) / len(y_pred_bin),
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "precision": precision_score(y_true, y_pred_bin),
        "recall": recall_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin),
        "auc": roc_auc_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred_bin),
    }
    return scores

def print_scores(y_true, y_pred, threshold_step=0.05, best_score_name="mcc"):
    threshold_range = np.arange(threshold_step, 1, threshold_step)
    # threshold_range = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999]

    score_names = {
        "ratio": "R",
        "accuracy": "ACC",
        "precision": "PREC",
        "recall": "REC",
        "f1": "F1",
        "auc": "AUC",
        "mcc": "MCC",
    }

    print("    |" + "|".join("{:>6s}".format(n) for n in score_names.values()))
    print("-"*(4+7*len(score_names)))
    string_scores = "{threshold:4.2f}|" + " ".join("{%s:6.3f}"%n for n in score_names)

    best_score = None
    best_score_threshold = None
    for threshold in threshold_range:
        scores = get_scores(y_true, y_pred, threshold=threshold)
        print(string_scores.format(threshold=threshold, **scores))

        if best_score is None or scores[best_score_name] > best_score:
            best_score = scores[best_score_name]
            best_score_threshold = threshold

    print()
    print("Best {:s} score: {:.4f}".format(best_score_name, best_score))
    print("Threshold: {:.3f}".format(best_score_threshold))

    return


def main(
        filename,
        threshold_step=0.05,
        score_name="mcc",
        ):

    data_predictions = pd.read_csv(filename)
    data_train_labels = pd.read_csv(filename_train_labels)

    names = data_predictions.filename.values.tolist()
    names = [str(name).replace(".npy", ".mp4") for name in names]
    predictions = data_predictions.prediction.values
    names = [os.path.basename(name) + ".mp4" for name in names]

    data_train_labels = data_train_labels[
        data_train_labels.filename.isin(names)]
    names_train = data_train_labels.filename.values.tolist()
    labels_train = data_train_labels.stalled.values

    indexes = [names_train.index(name) for name in names]
    # indexes = [names.index(name) for name in names_train]
    labels = labels_train[indexes]

    print(names[:5])
    print([names_train[index] for index in indexes][:5])
    print(names_train[:5])
    print(predictions[:5])
    print(labels[:5])

    print("{:5s}: {:5d} {:.4f}".format(
        "True", np.sum(labels), np.sum(labels) / len(labels)))
    print("{:5s}: {:.3f} {:.3f} {:.3f} {:.3f}".format(
        "Pred", np.min(predictions), np.max(predictions), 
        np.mean(predictions), np.std(predictions),))
    print()

    print_scores(labels, predictions, 
        threshold_step=threshold_step, best_score_name=score_name)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file")
    parser.add_argument("--threshold_step", help="step for range of threshold values",
        default=0.05, type=float)
    parser.add_argument("--score_name", help="name of score metric to look for best threshold",
        default="mcc")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file,
        threshold_step=args.threshold_step,
        score_name=args.score_name,
    )
