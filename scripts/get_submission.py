import os
import sys
import numpy as np
import pandas as pd
import argparse


__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_submissions = os.path.join(__folder_res, "submissions")
filename_test_metadata = os.path.join(__folder_data, "test_metadata.csv")

def binarize(predictions, threshold=0.5):
    return predictions >= threshold

def main(filename_predictions, filename_out, threshold=0.5):
    data_predictions = pd.read_csv(filename_predictions)
    names = data_predictions.filename.values.tolist()
    predictions = data_predictions.prediction.values
    if os.path.isfile(filename_test_metadata):
        data_test_metadata = pd.read_csv(filename_test_metadata)
        names_test = data_test_metadata.filename.values.tolist()
        indexes = [names.index(n) for n in names_test]
        names = [names[i] for i in indexes]
        predictions = predictions[indexes]
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
    parser.add_argument("--out", help="output submission file", 
        default=None)
    parser.add_argument("--threshold", help="threshold for true values",
        default=0.5, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.out is None:
        basename = os.path.basename(args.path)
        model_name = basename.split("-")[0]
        ep = basename.split("-")[1]
        args.out = os.path.join(__folder_submissions, "{:s}-{:s}-th{:.2f}.csv".format(
            model_name, ep, args.threshold))
    main(args.path, args.out, threshold=args.threshold)