import os
import numpy as np
import random
# from .processing.convert import ()


class DataLoader:
    def __init__(self, 
            folder=None,
            names=[],
            targets=None,
            extension=None,
            shuffle=False,    
            ):
        self.folder     = folder
        self.names      = names
        self.shuffle    = shuffle
        self.extension  = extension
        self.targets    = targets

    def get_filename(self, name):
        if isinstance(name, int):
            name = self.names[name]
        if self.extension:
            if self.extension[0] != ".":
                name += "." + self.extension
            else:
                name += self.extension
        if self.folder:
            return os.path.join(self.folder, name)
        else:
            return name
    def get_target(self, index):
        if isinstance(index, str):
            if index in self.names:
                index = self.names.index(index)
            else:
                return 0
        if self.targets is not None and len(self.targets) > 0:
            return self.targets[index]
        else:
            return 0

    def get_indexes(self, shuffle=None):
        if shuffle is None:
            shuffle = self.shuffle
        indexes = np.arange(len(self.names)).tolist()
        if shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __read(self, filename):
        return np.load(filename)
    def read(self, name):
        filename = self.get_filename(name)
        return self.__read(filename)
    def process(self, input):
        return input
    def load(self, name):
        return self.process(self.read(name))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return self.load(index), self.get_target(index)
    def __iter__(self):
        for index in self.get_indexes(shuffle=False):
            yield self.load(index), self.get_target(index)

    def generator(self, load_target=False):
        while True:
            indexes = self.get_indexes()
            for index in indexes:
                if load_target:
                    yield self.load(index), self.get_target(index)
                else:
                    yield self.load(index)

    def get_batch_number(self, batch_size=16, leave_last=True):
        if leave_last:
            return np.ceil(len(self.names) / batch_size)
        else:
            return np.floor(len(self.names) / batch_size)

    def batch_generator(self, batch_size=16, leave_last=True, load_target=False):
        while True:
            indexes = self.get_indexes()
            batch, targets = [], []
            for index in indexes:
                batch.append(self.load(index))
                targets.append(self.get_target(index))
                if len(batch) == batch_size:
                    if load_target:
                        yield batch, targets
                    else:
                        yield batch
                    batch, targets = [], []
            if leave_last and len(batch) > 0:
                if load_target:
                    yield batch, targets
                else:
                    yield batch


class DataLoaderReader(DataLoader):
    def __init__(self,
            shape=(32, 32, 32),
            augmentation=False, 
            extension=".npy", 
            **kwargs):
        super(DataLoaderReader, self).__init__(extension=extension, **kwargs)
        self.augmentation = augmentation
        self.shape = shape

    def process(self, input):
        if self.augmentation:
            if random.getrandbits(1):
                input[:] = input[::-1]
            if random.getrandbits(1):
                input[:] = input[:,::-1]
            if random.getrandbits(1):
                input[:] = input[:,:,::-1]
            return input
        else:
            return input

    def batch_generator(self, batch_size=16, leave_last=True, load_target=False):
        batch = np.empty((batch_size,)+self.shape, np.float32)
        targets = np.empty((batch_size,), np.float32)
        while True:
            indexes = self.get_indexes()
            index_inbatch = 0
            for index in indexes:
                batch[index_inbatch] = self.load(index)
                targets[index_inbatch] = self.get_target(index)
                index_inbatch += 1
                if index_inbatch == batch_size:
                    index_inbatch = 0
                    if load_target:
                        yield batch, targets
                    else:
                        yield batch
            if leave_last and index_inbatch:
                if load_target: 
                    yield batch[:index_inbatch], targets[:index_inbatch]
                else:
                    yield batch[:index_inbatch]

    