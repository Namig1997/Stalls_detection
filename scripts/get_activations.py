import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tensorflow as tf
import argparse
import json
from tqdm import tqdm


__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import DataLoaderReader, DataLoaderProcesser
from net.layers import _custom_objects
from net.model import get_simple_model

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
        out,
        batch_size=None,
        epoch=None, 
        ):
    
    folder_model = os.path.join(__folder_models, model_name)
    folder_checkpoints = os.path.join(folder_model, "checkpoints")

    filename_data_train = os.path.join(folder_model, "train.csv")
    filename_data_val   = os.path.join(folder_model, "test.csv")

    data_train  = pd.read_csv(filename_data_train)
    data_val    = pd.read_csv(filename_data_val)
    data_test   = pd.read_csv(filename_test_metadata)
    names_train = [n.split(".")[0] for n in data_train.filename.values.tolist()]
    names_val   = [n.split(".")[0] for n in data_val.filename.values.tolist()]
    names_test  = [n.split(".")[0] for n in data_test.filename.values.tolist()]


    filename_params = os.path.join(folder_model, "params.json")
    params = read_json(filename_params)
    if epoch is None:
        filename_checkpoint = os.path.join(folder_checkpoints, "best")
    else:
        filename_checkpoint = os.path.join(folder_checkpoints, "{:02d}".format(epoch))
    if batch_size is not None:
        params["batch_size"] = batch_size

    dataloader = DataLoaderProcesser(
        folder=folder_default_train,
        names=names_train,
        shape=params["shape"],
        normed_xy=params["normed_xy"],
    )

    # model = tf.keras.models.load_model(filename_checkpoint,
    #     custom_objects=_custom_objects, compile=True)
    model = get_simple_model(tuple(dataloader.shape), version=1)
    model._set_inputs(tf.keras.Input(dataloader.shape))
    model.load_weights(filename_checkpoint, by_name=True)

    model_2 = tf.keras.Model(
        inputs=model.input,
        outputs=[model.layers[-1].output, model.layers[-2].output],
    )
    os.makedirs(out, exist_ok=True)


    # TRAIN SET
    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])
    predictions, activations = model_2.predict(
        batch_generator, 
        steps=batch_number,
        verbose=1, 
        workers=0,
    )
    data_train_predictions = pd.DataFrame()
    data_train_predictions["filename"]      = data_train.filename
    data_train_predictions["stalled"]       = data_train.stalled
    data_train_predictions["prediction"]    = predictions
    for i in range(activations.shape[1]):
        # print("{:3d} {:6.3f} {:6.3f} {:6.3f}".format(
        #     i, activations[:, i].min(), 
        #     activations[:, i].max(), activations[:, i].mean()))
        data_train_predictions["activation_{:d}".format(i)] = activations[:, i]
    data_train_predictions.to_csv(os.path.join(out, "train.csv"), index=False)

    # VALIDATION SET
    dataloader.names = names_val
    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])
    predictions, activations = model_2.predict(
        batch_generator, 
        steps=batch_number,
        verbose=1, 
        workers=0,
    )
    data_val_predictions = pd.DataFrame()
    data_val_predictions["filename"]    = data_val.filename
    data_val_predictions["stalled"]     = data_val.stalled
    data_val_predictions["prediction"]  = predictions
    for i in range(activations.shape[1]):
        # print("{:3d} {:6.3f} {:6.3f} {:6.3f}".format(
        #     i, activations[:, i].min(), 
        #     activations[:, i].max(), activations[:, i].mean()))
        data_val_predictions["activation_{:d}".format(i)] = activations[:, i]
    data_val_predictions.to_csv(os.path.join(out, "validation.csv"), index=False)

    # TEST SET
    dataloader.names = names_test
    dataloader.folder = folder_default_test
    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])
    predictions, activations = model_2.predict(
        batch_generator, 
        steps=batch_number,
        verbose=1, 
        workers=0,
    )
    data_test_predictions = pd.DataFrame()
    data_test_predictions["filename"]   = data_test.filename
    data_test_predictions["prediction"] = predictions
    for i in range(activations.shape[1]):
        # print("{:3d} {:6.3f} {:6.3f} {:6.3f}".format(
        #     i, activations[:, i].min(), 
        #     activations[:, i].max(), activations[:, i].mean()))
        data_test_predictions["activation_{:d}".format(i)] = activations[:, i]
    data_test_predictions.to_csv(os.path.join(out, "test.csv"), index=False) 

    return   

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name",
        default=None)
    parser.add_argument("--out", help="output folder",
        default=None)
    parser.add_argument("--batch_size", help="batch size",
        default=None, type=int)
    parser.add_argument("--epoch", help="index of checkpoint to load; if not provided, best is loaded")
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

    if args.out is None:
        if args.epoch is not None:
            ep = "e{:02d}".format(args.epoch)
        else:
            ep = "bst"
        args.out = os.path.join(__folder_res, 
            "outputs", "{:s}-{:s}".format(args.model, ep))

    main(
        args.model,
        args.out, 
        batch_size=args.batch_size,
        epoch=args.epoch,
    )