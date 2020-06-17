import os
import sys
import numpy as np
import pandas as pd
import argparse



def binarize(predictions, threshold=0.5):
    return predictions >= threshold

def main(filename_predictions, filename_out, threshold=0.5):
    data_predictions = pd.read_csv(filename_predictions)
    names = data_predictions.filename.values.tolist()
    predictions = data_predictions.prediction.values
    print(predictions.mean(), predictions.std())
    predictions_bin = binarize(predictions, threshold=threshold)
    predictions_bin = np.array(predictions_bin, np.int8)
    print(len(predictions), np.sum(predictions_bin), 
        np.sum(predictions_bin) / len(predictions))
    data_submission = pd.DataFrame(
        {"filename": names, "stalled": predictions_bin}
    )
    data_submission.to_csv(filename_out, index=False)
    return predictions_bin

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to predictions file")
    parser.add_argument("out", help="output submission file")
    parser.add_argument("--threshold", help="threshold for true values",
        default=0.5, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.out, threshold=args.threshold)