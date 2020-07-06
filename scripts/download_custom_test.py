import os
import sys
import pandas as pd
import numpy as np
import argparse

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from download import Downloader


__folder_res = os.path.join(__folder, "res")
__folder_data = os.path.join(__folder_res, "data")
filename_train_metadata = os.path.join(__folder_data, "train_metadata.csv")


def main(
        folder,
        filename_list=None,
        filename_out=None,
        num=20000,
        seed=0,
        skip=True,
        ):

    if not filename_list:
        data_train_metadata = pd.read_csv(filename_train_metadata)
        # data_train_labels = pd.read_csv("./res/data/train_labels.csv")
        # names_stalled = data_train_labels[data_train_labels.stalled == 1].filename.values.tolist()
        # data_train_metadata = data_train_metadata[
        #     data_train_metadata.filename.isin(names_stalled)]
        # names = data_train_metadata.filename.values.tolist()
        # urls = data_train_metadata.url.values.tolist()

        if seed is not None:
            np.random.seed(seed)
        indexes = np.random.choice(np.arange(len(names)), num, replace=False)
        names = [names[i] for i in indexes]
        urls = [urls[i] for i in indexes]
    else:
        with open(filename_list, "r") as file:
            names = file.read().split("\n")
        for i in range(len(names)):
            if not names[i].endswith(".mp4"):
                names[i] += ".mp4"
        urls = ["s3://drivendata-competition-clog-loss/train/" + name for name in names]

    if filename_out:
        with open(filename_out, "w") as file:
            file.write("\n".join(names))

    Downloader().download_list(urls, folder)

    return names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder to save into")
    parser.add_argument("--list", help="path to file with list of names to download",
        default=None)
    parser.add_argument("--out", help="output file with list of downloaded names",
        default=None)
    parser.add_argument("--num", help="number of files to sample",
        default=20000, type=int)
    parser.add_argument("--seed", help="seed for numpy random",
        default=0, type=int)
    parser.add_argument("--no_skip", help="if set, existing files will be rewritten",
        action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.folder,
        filename_list=args.list,
        filename_out=args.out,
        num=args.num,
        seed=args.seed,
        skip=args.no_skip,
    )