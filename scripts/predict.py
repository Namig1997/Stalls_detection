import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tensorflow as tf
import argparse
import json

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import DataLoaderReader, DataLoaderProcesser
from net.layers import _custom_objects


__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_models = os.path.join(__folder_res, "models")
__folder_preds  = os.path.join(__folder_res, "predictions")
filename_test_metadata = os.path.join(__folder_data, "test_metadata.csv")
folder_default_train = os.path.join(__folder_data, "cut", "large_120000_cut")
folder_default_test  = os.path.join(__folder_data, "cut", "test_cut")


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def main(
        model_name,
        folder_data,
        out,
        batch_size=None,
        epoch=None,
        filename_csv=None,
        ): 

    folder_model = os.path.join(__folder_models, model_name)
    folder_checkpoints = os.path.join(folder_model, "checkpoints")

    if filename_csv:
        data = pd.read_csv(filename_csv)
        filenames = data.filename.values.tolist()
    else:
        filenames = [f.name.replace(".npy", ".mp4") for f in os.scandir(folder_data)]

    names = [n.split(".")[0] for n in filenames]

    filename_params = os.path.join(folder_model, "params.json")
    params = read_json(filename_params)
    if epoch is None:
        filename_checkpoint = os.path.join(folder_checkpoints, "best")
    else:
        filename_checkpoint = os.path.join(folder_checkpoints, "{:02d}".format(epoch))
    
    if batch_size is not None:
        params["batch_size"] = batch_size

    dataloader = DataLoaderProcesser(
        folder=folder_data,
        names=names,
        shape=params["shape"],
        normed_xy=params["normed_xy"],
    )

    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])

    model = tf.keras.models.load_model(filename_checkpoint, 
        custom_objects=_custom_objects, compile=False)

    predictions = model.predict(
        batch_generator, 
        steps=batch_number,
        verbose=1,
        workers=0,
    )
    predictions = np.reshape(predictions, (len(predictions),))

    data_predictions = pd.DataFrame(
        {
            "filename": filenames, 
            "prediction": predictions,
        }
    )
    data_predictions.to_csv(out, index=False)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name", 
        default=None)
    parser.add_argument("--data", help="path to data folder", 
        default=None)
    parser.add_argument("--out", help="output predictions csv file", 
        default=None)
    parser.add_argument("--batch_size", help="batch size",
        default=None, type=int)
    parser.add_argument("--epoch", help="index of checkpoint to load; if not provided, best is loaded",
        default=None, type=int)
    parser.add_argument("--csv", help="path to csv file to get filenames from",
        default=None)
    parser.add_argument("--set", help="default sets of filenames from training, validation or test sets", 
        choices=["train", "val", "test"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model is None:
        names = [f.name for f in os.scandir(__folder_models)]
        names_d = [n for n in names if 
            n[2:4].isdigit() and n[:2].isdigit() and n[5:].isdigit()]
        names_d.sort(key=lambda x: 
            int(x[2:4])*100*1000+int(x[:2])*1000+int(x[5:]))
        if len(names_d) > 0:
            args.model = names_d[-1]
        else:
            args.model = names[0]

    if args.data is None and args.set is not None:
        if args.set in ["train", "val"]:
            args.data = folder_default_train
        elif args.set in ["test"]:
            args.data = folder_default_test

    if args.model is None:
        print("Model is not set")
        quit()
    if args.data is None:
        print("Input data is not set")
        quit()

    if args.out is None:
        if args.epoch is not None:
            ep = "e{:02d}".format(args.epoch)
        else:
            ep = "bst"
        if args.set is not None:
            se = args.set
        else:
            se = "n"
        args.out = os.path.join(__folder_preds, "{:s}-{:s}-{:s}.csv".format(
            args.model, ep, se))

    if args.set is not None:
        if args.set == "test":
            args.csv = filename_test_metadata
        elif args.set == "train":
            args.csv = os.path.join(__folder_models, args.model, "train.csv")
        elif args.set == "val":
            args.csv = os.path.join(__folder_models, args.model, "test.csv")


    print("Model: {:s}".format(args.model))
    if args.csv:
        print("Data: {:s} {:s}".format(args.data, args.csv))
    else:
        print("Data: {:s}".format(args.data))
    print("Out: {:s}".format(args.out))


    main(
        args.model,
        args.data,
        args.out,
        batch_size=args.batch_size,
        epoch=args.epoch,
        filename_csv=args.csv,
    )