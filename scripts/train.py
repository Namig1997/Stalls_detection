import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
import argparse

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import DataLoaderReader
from net.model import get_simple_model, get_simple_model_2
from net.metrics import MCC
from net.logs import CustomLogger


__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_models = os.path.join(__folder_res, "models")
filename_train_labels = os.path.join(__folder_data, "train_labels.csv")


def _basename(folder):
    name = os.path.basename(folder)
    if len(name) == 0:
        name = os.path.basename(os.path.dirname(folder))
    return name


def main(
        model_name,
        folder_data,
        validation_size=0.1,
        batch_size=32,
        epochs=40,
        verbose=2,
        shape = (32, 32, 32),
        ):

    # read data from csv: names and labels
    data_train_labels = pd.read_csv(filename_train_labels)
    names = data_train_labels.filename.values.tolist()
    labels = data_train_labels.stalled.values

    names = [n.split(".")[0] for n in names]
    names_in_folder = [f.name.split(".")[0] for f in os.scandir(folder_data)]

    indexes_in_list = []
    for i, n in enumerate(names):
        if n in names_in_folder:
            indexes_in_list.append(i)
    
    labels = labels[indexes_in_list]
    names = [names[i] for i in indexes_in_list]

    # split
    indexes = np.arange(len(names))
    np.random.shuffle(indexes)
    indexes_train = indexes[:int(len(indexes)*(1-validation_size))]
    indexes_test = indexes[len(indexes_train):]

    print("{:6s} {:6d} {:.4f}".format(
        "Train:", len(indexes_train), 
        np.sum(labels[indexes_train]) / len(indexes_train)))
    print("{:6s} {:6d} {:.4f}".format(
        "Test:", len(indexes_test), 
        np.sum(labels[indexes_test]) / len(indexes_test)))

    data_train = pd.DataFrame(
        {
            "filename": [names[i] + ".mp4" for i in indexes_train],
            "stalled": labels[indexes_train],
        }
    )
    data_test = pd.DataFrame(
        {
            "filename": [names[i] + ".mp4" for i in indexes_test],
            "stalled": labels[indexes_test],
        }
    )


    # paths for folders and files
    folder_model        = os.path.join(__folder_models, model_name)
    folder_checkpoints  = os.path.join(folder_model, "checkpoints")
    folder_logs         = os.path.join(folder_model, "logs")
    os.makedirs(folder_model,       exist_ok=True)
    os.makedirs(folder_checkpoints, exist_ok=True)
    os.makedirs(folder_logs,        exist_ok=True)
    filename_train_set          = os.path.join(folder_model, "train.csv")
    filename_test_set           = os.path.join(folder_model, "test.csv")
    filename_checkpoint_best    = os.path.join(folder_checkpoints, "best")
    filename_checkpoint         = os.path.join(folder_checkpoints, "{epoch:02d}")
    filename_train_preds        = os.path.join(folder_model, "train_pred.csv")
    filename_test_preds         = os.path.join(folder_model, "test_pred.csv")
    data_train.to_csv(filename_train_set,   index=False)
    data_test.to_csv(filename_test_set,     index=False)


    # init and compile model
    # model = get_simple_model(shape)
    model = get_simple_model_2(shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"), 
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            MCC(name="mcc"),
        ],
    )

    # init dataloder objects for train and test data
    dataloader_train = DataLoaderReader(
        folder=folder_data,
        names=[names[i] for i in indexes_train],
        targets=labels[indexes_train],
        shape=shape,
        shuffle=True,
        augmentation=True,
    )
    dataloader_test = DataLoaderReader(
        folder=folder_data,
        names=[names[i] for i in indexes_test],
        targets=labels[indexes_test],
        shape=shape,
    )

    # get batch generators for train and test data
    batch_number_train = dataloader_train.get_batch_number(batch_size, False)
    batch_number_test  = dataloader_test.get_batch_number(batch_size)
    batch_generator_train = dataloader_train.batch_generator(batch_size, 
        leave_last=False, load_target=True)
    batch_generator_test  = dataloader_test.batch_generator(batch_size, load_target=True)

    # object for outputting and logging model losses and metrics while training
    custom_logger = CustomLogger(
        verbose=verbose, 
        model_name=_basename(folder_model),
        folder=folder_logs, 
        names=[
            [
                "accuracy",
                "precision",
                "recall",
                "auc",
                "mcc",
            ]
        ],
    )

    # all callbacks
    callbacks = [
        # reduction on plateau of the learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", 
            patience=10, 
            mode="max",
        ),
        # save all epochs
        tf.keras.callbacks.ModelCheckpoint(
            filename_checkpoint),
        # save best epoch
        tf.keras.callbacks.ModelCheckpoint(
            filename_checkpoint_best,
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
        # logs
        custom_logger,
    ]

    # run training
    model.fit(
        batch_generator_train,
        steps_per_epoch=batch_number_train*2,
        validation_data=batch_generator_test,
        validation_steps=batch_number_test,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # reload weights of best model epoch
    model.load_weights(filename_checkpoint_best)


    # running final predictions for train data
    dataloader = DataLoaderReader(
        folder=folder_data,
        names=[names[i] for i in indexes_train],
        shape=shape,
    )
    batch_number = dataloader.get_batch_number(batch_size)
    batch_generator = dataloader.batch_generator(batch_size)
    predictions = model.predict(
        batch_generator,
        steps=batch_number,
        verbose=1, 
        workers=0,
    )
    data_train["prediction"] = np.reshape(predictions, (len(predictions,)))
    data_train.to_csv(filename_train_preds, index=False)

    # running final predictions for test data
    dataloader = DataLoaderReader(
        folder=folder_data,
        names=[names[i] for i in indexes_test],
        shape=shape,
    )
    batch_number = dataloader.get_batch_number(batch_size)
    batch_generator = dataloader.batch_generator(batch_size)
    predictions = model.predict(
        batch_generator,
        steps=batch_number,
        verbose=1,
        workers=0,
    )
    data_test["prediction"] = np.reshape(predictions, len(predictions,))
    data_test.to_csv(filename_test_preds, index=False)

    return



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("folder", help="path to data folder")
    parser.add_argument("--validation_size", help="ratio size of validation split",
        default=0.1, type=float)
    parser.add_argument("--batch_size", help="batch size", 
        default=32, type=int)
    parser.add_argument("--epochs", help="number of train epochs",
        default=40, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model, 
        args.folder,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        )