import os
import sys
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
from models.layers import *


__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_models = os.path.join(__folder_res, "models")
filename_train_labels = os.path.join(__folder_data, "train_labels.csv")

sys.path.append(os.path.join(__folder, "scripts"))
from get_metrics import print_scores

def get_simple_model(input_shape=(32, 32, 32)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape + (1,)),
        ConvBlock_1(32),
        ConvBlock_1(64),
        ConvBlock_1(64),
        ConvBlock_1(64),
        Conv3D_bn(64, kernel_size=2, padding="valid"),
        tf.keras.layers.Reshape((64,)),
        Dense(1, activation=tf.keras.activations.sigmoid),
    ])
    return model


def main(
        model_name,
        folder_data,
        validation_size=0.2,
        batch_size=32,
        epochs=40,
        ):
    
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

    indexes = np.arange(len(names))
    np.random.shuffle(indexes)
    indexes_train = indexes[:int(len(indexes)*(1-validation_size))]
    indexes_test = indexes[len(indexes_train):]

    print("Train:", len(indexes_train), np.sum(labels[indexes_train]) / len(indexes_train))
    print("Test:", len(indexes_test), np.sum(labels[indexes_test]) / len(indexes_test))

    shape = (32, 32, 32)
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

    batch_number_train = dataloader_train.get_batch_number(batch_size, False)
    batch_number_test  = dataloader_test.get_batch_number(batch_size)
    batch_generator_train = dataloader_train.batch_generator(batch_size, 
        leave_last=False, load_target=True)
    batch_generator_test  = dataloader_test.batch_generator(batch_size, 
        leave_last=True, load_target=True)


    folder_model = os.path.join(__folder_models, model_name)
    os.makedirs(folder_model, exist_ok=True)
    folder_checkpoints = os.path.join(folder_model, "checkpoints")
    os.makedirs(folder_checkpoints, exist_ok=True)


    model = get_simple_model(shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc", factor=0.5, patience=5, min_lr=1e-6,),
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(folder_checkpoints, "{epoch:02d}")),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(folder_checkpoints, "best"),
            save_best_only=True,
            monitor="val_acc",
            mode="max",),
    ]

    model.fit(
        batch_generator_train,
        steps_per_epoch=batch_number_train,
        validation_data=batch_generator_test,
        validation_steps=batch_number_test,
        epochs=epochs,
        callbacks=callbacks,
    )


    # dataloader = DataLoaderReader(
    #     folder=folder_data,
    #     names=names,
    #     shape=shape,
    # )
    # batch_number = dataloader.get_batch_number(batch_size)
    # batch_generator = dataloader.batch_generator(batch_size)

    # model.evaluate(batch_generator_train,  
    #     steps=batch_number_train, verbose=1)
    # model.evaluate(batch_generator_test,
    #     steps=batch_number_test, verbose=1)

    # predictions = model.predict(batch_generator, steps=batch_number, verbose=1)
    # print_scores(labels, predictions)
    # predictions = np.reshape(predictions, (len(predictions),))
    # filenames = [n.replace(".npy", "mp4") for n in names]
    # filenames_ = []
    # for n in filenames:
    #     if not n.endswith(".mp4"):
    #         n += ".mp4"
    #     filenames_.append(n)
    # data_predictions = pd.DataFrame(
    #     {"filename": filenames_, "prediction": predictions},
    # )
    # data_predictions.to_csv("./res/pred_temp.csv", index=False)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("folder", help="path to data folder")
    parser.add_argument("--validation_size", help="ratio size of validation split",
        default=0.2, type=float)
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