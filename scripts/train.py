import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import json
import datetime
from zipfile import ZipFile

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import DataLoaderReader, DataLoaderProcesser
from net.model import get_simple_model
from net.metrics import MCC
from net.logs import CustomLogger
from net.callback import ReduceLROnPlateauRestore
from net.layers import _custom_objects
_custom_objects["MCC"] = MCC

__folder_scripts = os.path.join(__folder, "scripts")
__folder_res    = os.path.join(__folder, "res")
__folder_data   = os.path.join(__folder_res, "data")
__folder_models = os.path.join(__folder_res, "models")
filename_train_labels = os.path.join(__folder_data, "train_labels.csv")


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _basename(folder):
    name = os.path.basename(folder)
    if len(name) == 0:
        name = os.path.basename(os.path.dirname(folder))
    return name

def write_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    return json.dumps(data)

def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def zipfiles(filenames_in, filename_out):
    obj = ZipFile(filename_out, "w")
    for filename in filenames_in:
        obj.write(filename)
    obj.close()
    return

def get_model_name(folder=__folder_models):
    names = [f.name for f in os.scandir(folder)]
    time = datetime.datetime.now()
    prefix = time.strftime("%d%m")
    max_index = -1
    for name in names:
        if name[:4] == prefix:
            max_index = max(int(name[5:]), max_index)
    return "{:s}_{:d}".format(prefix, max_index+1)
    

default_params = {
    "shape": (32, 32, 32),
    "shuffle": True,
    "balanced": True,
    "class_ratio": 10,
    "augmentation": True,
    "size_change_z": 0.05,
    "size_change_xy": 0.05,
    "shift_z": 0.05,
    "shift_xy": 0.05,
    "rotation": False,
    "rotation_z": True,
    "normed_xy": False,
    "noise": 0.,
    "monitor_metric": "mcc_09",
    "monitor_mode": "max",
    "learning_rate_start": 1e-3,
    "learning_rate_factor": 0.1,
    "learning_rate_patience": 10,
    "learning_rate_min": 1e-6,
    "batch_size": 32,
    "epochs": 100,
    "steps": None,
    "steps_mult": None,
    "early_stopping_patience": 40,
    "model_version": 1, 
}

def adjust_params(params, args):
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.shape is not None:
        params["shape"] = (args.shape, args.shape, args.shape)
    if args.shape_z is not None:
        params["shape"] = (args.shape_z, params["shape"][1], params["shape"][2])
    if args.shape_xy is not None:
        params["shape"] = (params["shape"][0], args.shape_xy, args.shape_xy)
    
    if args.shuffle is not None:
        params["shuffle"] = bool(args.shuffle)
    if args.balanced is not None:
        params["balanced"] = bool(args.balanced)
    if args.augmentation is not None:
        params["augmentation"] = bool(args.augmentation)
    if args.size_change_z is not None:
        params["size_change_z"] = args.size_change_z
    if args.size_change_xy is not None:
        params["size_change_xy"] = args.size_change_xy
    if args.shift_z is not None:
        params["shift_z"] = args.shift_z
    if args.shift_xy is not None:
        params["shift_xy"] = args.shift_xy
    if args.rotation is not None:
        params["rotation"] = bool(args.rotation)
    if args.rotation_z is not None:
        params["rotation_z"] = bool(args.rotation_z)
    if args.noise is not None:
        params["noise"] = args.noise
    if args.normed_xy is not None:
        params["normed_xy"] = bool(args.normed_xy)

    if args.monitor_metric is not None:
        params["monitor_metric"] = args.monitor_metric
    if args.monitor_mode is not None:
        params["monitor_mode"] = args.monitor_mode
    if args.learning_rate_start is not None:
        params["learning_rate_start"] = args.learning_rate_start
    if args.learning_rate_factor is not None:
        params["learning_rate_factor"] = args.learning_rate_factor
    if args.learning_rate_patience is not None:
        params["learning_rate_patience"] = args.learning_rate_patience
    if args.learning_rate_min is not None:
        params["learning_rate_min"] = args.learning_rate_min
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.steps is not None:
        params["steps"] = args.steps
    if args.steps_mult is not None:
        params["steps_mult"] = args.steps_mult
    if args.early_stopping_patience is not None:
        params["early_stopping_patience"] = args.early_stopping_patience
    if args.model_version is not None:
        params["model_version"] = args.model_version

    return params


def main(
        model_name,
        folder_data,
        validation_size=0.1,
        train_set=None,
        validation_set=None,
        verbose=2,
        random_seed=0,
        load_model=None, 
        load_epoch=None,
        **params,
        ):

    try: 
        tf.random.set_seed(random_seed)
    except:
        tf.compat.v1.set_random_seed(random_seed)
    np.random.seed(random_seed)

    # read data from csv: names and labels
    data_train_labels = pd.read_csv(filename_train_labels)
    names = data_train_labels.filename.values.tolist()
    labels = data_train_labels.stalled.values

    names = [n.split(".")[0] for n in names]
    names_in_folder = [f.name.split(".")[0] for f in os.scandir(folder_data)]
    names_in_folder_s = set(names_in_folder)

    indexes_in_list = []
    for i, n in enumerate(names):
        if n in names_in_folder_s:
            indexes_in_list.append(i)
    
    labels = labels[indexes_in_list]
    names = [names[i] for i in indexes_in_list]

    
    # split
    indexes = np.arange(len(names))
    np.random.shuffle(indexes)
    indexes_train = indexes[:int(len(indexes)*(1-validation_size))]
    indexes_test = indexes[len(indexes_train):]
    if validation_set:
        data_val_set = pd.read_csv(validation_set)
        names_val = data_val_set.filename.values.tolist()
        names_val = [n.split(".")[0] for n in names_val]
        names_val_s = set(names_val)
        indexes_test = [i for i in np.arange(len(names)) if names[i] in names_val_s]
        indexes_train = [i for i in np.arange(len(names)) if i not in indexes_test]
    if train_set:
        data_train_set = pd.read_csv(train_set)
        names_train = data_train_set.filename.values.tolist()
        names_train = [n.split(".")[0] for n in names_train]
        names_train_s = set(names_train)
        indexes_train = [i for i in np.arange(len(names)) if names[i] in names_train_s]

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
    filename_params             = os.path.join(folder_model, "params.json")
    filename_zip                = os.path.join(folder_model, "code.zip")
    data_train.to_csv(filename_train_set,   index=False)
    data_test.to_csv(filename_test_set,     index=False)
    write_json(params, filename_params)
    zipfiles([__folder_code, __folder_scripts], filename_zip)


    # init and compile model
    model = get_simple_model(params["shape"], 
            version=params["model_version"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate_start"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"), 
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            MCC(threshold=0.1, name="mcc_01"),
            MCC(threshold=0.5, name="mcc_05"),
            MCC(threshold=0.9, name="mcc_09"),
            MCC(threshold=0.99, name="mcc_099"),
        ],
    )
    model.build((None,) + tuple(params["shape"]))

    if load_model:
        if load_epoch is not None:
            filename_checkpoint_pre = os.path.join(load_model, "checkpoints", 
                "{:02d}".format(load_epoch))
        else:
            filename_checkpoint_pre = os.path.join(load_model, "checkpoints", "best")
        # model = tf.keras.models.load_model(filename_checkpoint_pre,
        #     custom_objects=_custom_objects, compile=True)
        model.load_weights(filename_checkpoint_pre)


    # init dataloder objects for train and test data
    dataloader_train = DataLoaderProcesser(
        folder=folder_data,
        names=[names[i] for i in indexes_train],
        targets=labels[indexes_train],
        shape           = params["shape"],
        shuffle         = params["shuffle"],
        balanced        = params["balanced"],
        class_ratio     = params["class_ratio"],
        augmentation    = params["augmentation"],
        size_change_z   = params["size_change_z"],
        size_change_xy  = params["size_change_xy"],
        shift_z         = params["shift_z"],
        shift_xy        = params["shift_xy"],
        rotation        = params["rotation"],
        noise           = params["noise"],
        normed_xy       = params["normed_xy"],
    )
    dataloader_test = DataLoaderProcesser(
        folder=folder_data,
        names=[names[i] for i in indexes_test],
        targets=labels[indexes_test],
        shape=params["shape"],
        normed_xy=params["normed_xy"],
    )

    # get batch generators for train and test data
    batch_number_train = dataloader_train.get_batch_number(params["batch_size"], False)
    batch_number_test  = dataloader_test.get_batch_number(params["batch_size"])
    batch_generator_train = dataloader_train.batch_generator(params["batch_size"], 
        leave_last=False, load_target=True)
    batch_generator_test  = dataloader_test.batch_generator(params["batch_size"], load_target=True)

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
            ],
            [
                "mcc_01",
                "mcc_05",
                "mcc_09",
                "mcc_099",
            ]
        ],
    )

    # all callbacks
    callbacks = [
        # reduction on plateau of the learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
        # ReduceLROnPlateauRestore(
            monitor = "val_{:s}".format(params["monitor_metric"]),
            factor  = params["learning_rate_factor"], 
            patience= params["learning_rate_patience"], 
            min_lr  = params["learning_rate_min"],
            mode    = params["monitor_mode"],
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor = "val_{:s}".format(params["monitor_metric"]),
            patience= params["early_stopping_patience"], 
            mode    = params["monitor_mode"],
        ),
        # save all epochs
        tf.keras.callbacks.ModelCheckpoint(
            filename_checkpoint,
            save_format="h5",
        ),
        # save best epoch
        tf.keras.callbacks.ModelCheckpoint(
            filename_checkpoint_best,
            save_best_only=True,
            monitor="val_{:s}".format(params["monitor_metric"]),
            mode=params["monitor_mode"],
            save_format="h5",
        ),
        # logs
        custom_logger,
    ]

    if params["steps"] is not None:
        steps_per_epoch = params["steps"]
    elif params["steps"] is not None:
        steps_per_epoch = int(batch_number_train * params["steps_mult"])
    else:
        steps_per_epoch = batch_number_train
    print("Batch number: {:d}/{:d} {:d}".format(
        batch_number_train, steps_per_epoch, batch_number_test))

    # run training
    model.fit(
        batch_generator_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=batch_generator_test,
        validation_steps=batch_number_test,
        epochs=params["epochs"],
        callbacks=callbacks,
        verbose=0,
        workers=0,
    )

    # reload weights of best model epoch
    model.load_weights(filename_checkpoint_best)


    # running final predictions for train data
    dataloader = DataLoaderProcesser(
        folder=folder_data,
        names=[names[i] for i in indexes_train],
        shape=params["shape"],
    )
    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])
    predictions = model.predict(
        batch_generator,
        steps=batch_number,
        verbose=1, 
        workers=0,
    )
    data_train["prediction"] = np.reshape(predictions, (len(predictions,)))
    data_train.to_csv(filename_train_preds, index=False)

    # running final predictions for test data
    dataloader = DataLoaderProcesser(
        folder=folder_data,
        names=[names[i] for i in indexes_test],
        shape=params["shape"],
    )
    batch_number = dataloader.get_batch_number(params["batch_size"])
    batch_generator = dataloader.batch_generator(params["batch_size"])
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
    parser.add_argument("--model", help="model name", 
        default=None)
    parser.add_argument("--data", help="path to data folder",
        default=None)
    parser.add_argument("--validation_size", help="ratio size of validation split",
        default=0.1, type=float)
    parser.add_argument("--train_set", help="path to csv file with samples for training",
        default=None)
    parser.add_argument("--validation_set", help="path to csv with filenames for validation",
        default=None)
    parser.add_argument("--load_model", help="if specified, the pretrained model will be loaded",
        default=None)
    parser.add_argument("--load_epoch", help="if set, model from corresponding checkpoint will be loaded",
        default=None)
    
    group_general = parser.add_argument_group("general")
    group_general.add_argument("--batch_size", help="batch size",
        default=None, type=int)
    group_general.add_argument("--shape", help="size of all dimensions",
        default=None, type=int)
    group_general.add_argument("--shape_xy", help="size of x,y [1, 2] dimensions",
        default=None, type=int)
    group_general.add_argument("--shape_z", help="size of z [0] dimension",
        default=None, type=int)
    
    group_dataset = parser.add_argument_group("dataset")
    group_dataset.add_argument("--shuffle", help="if 1, samples are shuffled each epoch",
        default=None, type=int, choices=[0, 1])
    group_dataset.add_argument("--balanced", help="if 1, classes are sampled equally",
        default=None, type=int, choices=[0, 1])
    group_dataset.add_argument("--augmentation", help="if 1, augmentations to input are applied during training",
        default=None, type=int, choices=[0, 1])
    group_dataset.add_argument("--size_change_z", help="std of change of size of input grid in z [0] dimension",
        default=None, type=float)
    group_dataset.add_argument("--size_change_xy", help="std of change of size of input grid in xy [0,1] dimensions",
        default=None, type=float)
    group_dataset.add_argument("--shift_z", help="std of shift of grid center in z [0] dimension",
        default=None, type=float)
    group_dataset.add_argument("--shift_xy", help="std of shift of grid center in xy [1,2] dimensions",
        default=None, type=float)
    group_dataset.add_argument("--rotation", help="if 1, random rotations are applied during interpolation",
        default=None, type=int, choices=[0, 1])
    group_dataset.add_argument("--rotation_z", help="if 1, random rotations around z [0] axis are applied",
        default=None, type=int, choices=[0, 1])
    group_dataset.add_argument("--noise", help="std of noise added to input",
        default=None, type=float)
    group_dataset.add_argument("--normed_xy", help="if 1, both xy dimensions [0,1] normed by the same shape",
        default=None, type=int, choices=[0, 1])

    group_training = parser.add_argument_group("training")
    group_training.add_argument("--monitor_metric", help="name of metric to monitor (for ReduceLROnPlateau,EarlyStopping,Checkpoints)",
        default="mcc_09", choices=["accuracy", "precision", "recall", "auc", "mcc_05", "mcc_09", "mcc_099"])
    group_training.add_argument("--monitor_mode", help="monitor mode (e.g. max for acc,prec,rec,auc,mcc and min for crossentropy",
        default="max", choices=["min", "max"])
    group_training.add_argument("--learning_rate_start", help="initial value of learning rate",
        default=None, type=float)
    group_training.add_argument("--learning_rate_factor", help="factor for ReduceLROnPlateau",
        default=None, type=float)
    group_training.add_argument("--learning_rate_patience", help="patience for ReduceLROnPlateau",
        default=None, type=int)
    group_training.add_argument("--learning_rate_min", help="minimum learning rate for ReduceLROnPlateau",
        default=None, type=float)
    group_training.add_argument("--epochs", help="number of train epochs",
        default=None, type=int)
    group_training.add_argument("--steps", help="number of steps per epoch, ",
        default=None, type=int)
    group_training.add_argument("--steps_mult", help="if specified, number of steps is calclated as number of batches in dataset * steps_mult",
        default=None, type=float)
    group_training.add_argument("--early_stopping_patience", help="patience for EarlyStopping",
        default=None, type=int)
    group_training.add_argument("--model_version", help="model architecture type",
        default=None, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model is None:
        args.model = get_model_name()
    if args.data is None:
        args.data = os.path.join(__folder_data, "cut", "large_120000_cut")

    if args.load_model and not os.path.isdir(args.load_model):
        args.load_model = os.path.join(__folder_models, args.load_model)

    if not args.load_model:
        params = {}
        params.update(default_params)
    else:
        params = read_json(os.path.join(args.load_model, "params.json"))
    adjust_params(params, args)

    main(
        args.model, 
        args.data,
        validation_size=args.validation_size,
        train_set=args.train_set,
        validation_set=args.validation_set,
        load_model=args.load_model,
        load_epoch=args.load_epoch,
        **params,
        )