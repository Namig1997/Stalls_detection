import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import datetime



class CustomLogger(tf.keras.callbacks.Callback):
    def __init__(self, 
            verbose=1, 
            names=[], 
            folder="./logs/", 
            append=False,
            model_name="",
            write_batch=True):
        self.verbose = verbose
        self.time_start = datetime.datetime.now()
        self.write_batch = write_batch
        self.model_name = model_name
        self.string_print = "{model_name:40s} {time:s} {mode:1s}{epoch:3d} {batch:>4s} {loss:6.3f}"
        self.names = []
        for name_group in names:
            self.string_print += " |"
            for name in name_group:
                self.names.append(name)
                self.string_print += " {" + name + ":5.3f}"
        self.string_print += " "
        self.epoch = 0
        self.create_folder(folder)
        if append:
            self.mode = "a"
        else:
            self.mode = "w"
        super(CustomLogger, self).__init__()

    def create_folder(self, folder):
        self.folder = folder
        self.filename_batch_train = os.path.join(self.folder, "batch_train.csv")
        self.filename_batch_test  = os.path.join(self.folder, "batch_test.csv")
        self.filename_epoch_train = os.path.join(self.folder, "epoch_train.csv")
        self.filename_epoch_test  = os.path.join(self.folder, "epoch_test.csv")
        self.filename_batch_formatted = os.path.join(self.folder, "batch_formatted.log")
        self.filename_epoch_formatted = os.path.join(self.folder, "epoch_formatted.log")
        os.makedirs(self.folder, exist_ok=True)
        self.file_batch_train = None
        self.file_batch_test  = None
        self.file_epoch_train = None
        self.file_epoch_test  = None
        self.file_batch_formatted = None
        self.file_epoch_formatted = None
        self.args_batch_train = ["time", "epoch", "batch", "loss"] + self.names
        self.args_batch_test  = ["time", "epoch", "batch", "loss"] + self.names
        self.args_epoch_train = ["time", "epoch", "loss"] + self.names
        self.args_epoch_test  = ["time", "epoch", "loss"] + self.names
        self.sep = ","

    def get_time(self):
        time_now = datetime.datetime.now()
        return time_now - self.time_start
    def get_time_string(self):
        time = self.get_time().total_seconds()
        h, r = divmod(time, 3600)
        m, s = divmod(r, 60)
        return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s))

    def adjust_epoch(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        else:
            self.epoch = epoch
        return epoch

    def get_args(self, epoch=None, batch=None, logs=None, get_val=False, mode=""):
        if "predictions" in logs:
            print(logs["predictions"])
        args = {}
        args["model_name"] = self.model_name
        args["time"]    = self.get_time_string()
        args["epoch"]   = self.adjust_epoch(epoch)
        if batch is None:
            args["batch"] = ""
        else:
            args["batch"]   = str(batch)
        args["mode"]    = mode
        for name in ["loss"] + self.names:
            if get_val:
                args[name] = logs["val_" + name]
            else:
                args[name] = logs[name]
        return args

    def start_file_batch_train(self):
        if self.file_batch_train is None:
            self.file_batch_train = open(self.filename_batch_train, self.mode)
            self.file_batch_train.write(self.sep.join(self.args_batch_train) + "\n")
    def start_file_batch_test(self):
        if self.file_batch_test is None:
            self.file_batch_test = open(self.filename_batch_test, self.mode)
            self.file_batch_test.write(self.sep.join(self.args_batch_test) + "\n")
    def start_file_epoch_train(self):
        if self.file_epoch_train is None:
            self.file_epoch_train = open(self.filename_epoch_train, self.mode)
            self.file_epoch_train.write(self.sep.join(self.args_epoch_train) + "\n")
    def start_file_epoch_test(self):
        if self.file_epoch_test is None:
            self.file_epoch_test = open(self.filename_epoch_test, self.mode)
            self.file_epoch_test.write(self.sep.join(self.args_epoch_test) + "\n")

    def start_file_batch_formatted(self):
        if self.file_batch_formatted is None:
            self.file_batch_formatted = open(self.filename_batch_formatted, self.mode)
    def start_file_epoch_formatted(self):
        if self.file_epoch_formatted is None:
            self.file_epoch_formatted = open(self.filename_epoch_formatted, self.mode)
            
    def write_batch_train(self, args):
        self.start_file_batch_train()
        self.file_batch_train.write(self.sep.join([str(args[n]) for n in self.args_batch_train]) + "\n")
    def write_batch_test(self, args):
        self.start_file_batch_test()
        self.file_batch_test.write(self.sep.join([str(args[n]) for n in self.args_batch_test]) + "\n")
    def write_epoch_train(self, args):
        self.start_file_epoch_train()
        self.file_epoch_train.write(self.sep.join([str(args[n]) for n in self.args_epoch_train]) + "\n")
    def write_epoch_test(self, args):
        self.start_file_epoch_test()
        self.file_epoch_test.write(self.sep.join([str(args[n]) for n in self.args_epoch_test]) + "\n")

    def write_batch_formatted(self, string):
        self.start_file_batch_formatted()
        self.file_batch_formatted.write(string + "\n")
    def write_epoch_formatted(self, string):
        self.start_file_epoch_formatted()
        self.file_epoch_formatted.write(string + "\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        args = self.get_args(batch=batch, logs=logs, mode="t")
        string = self.string_print.format(**args)
        if self.verbose >= 2:
            print(string, end="\r", flush=True)
        if self.write_batch:
            self.write_batch_train(args)
            self.write_batch_formatted(string)
        

    def on_test_batch_end(self, batch, logs=None):
        args = self.get_args(batch=batch, logs=logs, mode="v")
        string = self.string_print.format(**args)
        if self.verbose >= 2:
            print(string, end="\r", flush=True)
        if self.write_batch:
            self.write_batch_test(args)
            self.write_batch_formatted(string)

    def on_epoch_end(self, epoch, logs=None):
        args = self.get_args(epoch=epoch, logs=logs)
        self.write_epoch_train(args)
        args = self.get_args(epoch=epoch, logs=logs, get_val=True)
        self.write_epoch_test(args)
        args["loss"] = logs["loss"]
        string = self.string_print.format(**args)
        if self.verbose >= 1:
            print(string)
        self.write_epoch_formatted(string)