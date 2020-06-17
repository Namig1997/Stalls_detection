import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import DataLoaderReader
from models.layers import _custom_objects


__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_models = os.path.join(__folder_res, "models")
filename_test_metadata = os.path.join(__folder_data, "test_metadata.csv")

def main(
        name,
        folder_data,
        out,
        batch_size=32,
        test=False,
        ): 

    if test:
        data_test_metadata = pd.read_csv(filename_test_metadata)
        filenames = data_test_metadata.filename.values.tolist()
    else:
        filenames = [f.name.replace(".npy", ".mp4") for f in os.scandir(folder_data)]

    names = [n.split(".")[0] for n in filenames]

    shape = (32, 32, 32)
    dataloader = DataLoaderReader(
        folder=folder_data,
        names=names,
        shape=shape,
    )

    batch_number = dataloader.get_batch_number(batch_size)
    batch_generator = dataloader.batch_generator(batch_size)

    folder_model = os.path.join(__folder_models, name)
    folder_checkpoints = os.path.join(folder_model, "checkpoints")

    filename_checkpoint_best = os.path.join(folder_checkpoints, "best")
    model = tf.keras.models.load_model(filename_checkpoint_best, 
        custom_objects=_custom_objects, compile=False)

    predictions = model.predict(
        batch_generator, 
        steps=batch_number,
        verbose=1,
    )
    predictions = np.reshape(predictions, (len(predictions),))

    data_predictions = pd.DataFrame(
        {"filename": filenames, "prediction": predictions}
    )
    data_predictions.to_csv(out, index=False)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("folder", help="path to data folder")
    parser.add_argument("out", help="output predictions csv file")
    parser.add_argument("--batch_size", help="batch size",
        default=32, type=int)
    parser.add_argument("--test", help="if set, predictions are retrieved for the test data, i.e. files in the test_metadata.csv",
        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model,
        args.folder,
        args.out,
        batch_size=args.batch_size,
    )