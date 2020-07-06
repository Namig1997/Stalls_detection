import os
import sys
import numpy as np
import random

__folder_current = os.path.abspath(os.path.dirname(__file__))
sys.path.append(__folder_current)
from processing.convert import convert_video_to_occupancy
from processing.interpolation import (
    InterpolatorRegularGrid, InterpolatorCustom)


class DataLoader:
    def __init__(self, 
            folder=None,
            names=[],
            targets=None,
            extension=None,
            shuffle=False,
            balanced=False,
            ):
        self.folder     = folder
        self.names      = names
        self.shuffle    = shuffle
        self.balanced   = balanced
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

    def get_indexes(self, shuffle=None, balanced=None):
        if shuffle is None:
            shuffle = self.shuffle
        if balanced is None:
            balanced = self.balanced
        if balanced:
            indexes_0 = np.where(self.targets == 0)[0].tolist()
            indexes_1 = np.where(self.targets == 1)[0].tolist()
            indexes = []
            index_0, index_1 = 0, 0
            for i in range(max(len(indexes_0), len(indexes_1))):
                if index_0 == 0 and shuffle:
                    np.random.shuffle(indexes_0)
                if index_1 == 0 and shuffle:
                    np.random.shuffle(indexes_1)
                indexes.append(indexes_0[index_0])
                indexes.append(indexes_1[index_1])
                index_0 += 1
                index_1 += 1
                if index_0 == len(indexes_0):
                    index_0 = 0
                if index_1 == len(indexes_1):
                    index_1 = 0
        else:
            indexes = np.arange(len(self.names)).tolist()
            if shuffle:
                np.random.shuffle(indexes)
        return indexes

    def __read(self, filename):
        return np.load(filename)
    def read(self, name):
        filename = self.get_filename(name)
        return self.__read(filename)
    def assign(self, input, grid):
        grid[:] = input
    def process(self, input):
        return input
    def load(self, name):
        return self.process(self.read(name))
    def to_batch(self, name, grid):
        self.assign(self.read(name), grid)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return self.load(index), self.get_target(index)
    def __iter__(self):
        for index in self.get_indexes(shuffle=False, balanced=False):
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
        num = len(self.get_indexes())
        if leave_last:
            return int(np.ceil(num / batch_size))
        else:
            return int(np.floor(num / batch_size))

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

    # def process(self, input):
    #     if self.augmentation:
    #         if random.getrandbits(1):
    #             input[:] = input[::-1]
    #         if random.getrandbits(1):
    #             input[:] = input[:,::-1]
    #         if random.getrandbits(1):
    #             input[:] = input[:,:,::-1]
    #         return input
    #     else:
    #         return input

    def assign(self, input, grid):
        grid[:] = input
        if self.augmentation:
            if random.getrandbits(1):
                grid[:] = grid[::-1]
            if random.getrandbits(1):
                grid[:] = grid[:,::-1]
            if random.getrandbits(1):
                grid[:] = grid[:,:,::-1]
    def process(self, input):
        grid = np.empty(self.shape, np.float32)
        self.assign(input, grid)
        return grid


    def batch_generator(self, batch_size=16, leave_last=True, load_target=False):
        batch = np.empty((batch_size,)+tuple(self.shape), np.float32)
        targets = np.empty((batch_size,), np.float32)
        while True:
            indexes = self.get_indexes()
            index_inbatch = 0
            for index in indexes:
                # batch[index_inbatch] = self.load(index)
                self.to_batch(index, batch[index_inbatch])
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



class DataLoaderProcesser(DataLoaderReader):
    def __init__(self,
            size_change_z=0.1,
            size_change_xy=0.1,
            shift_z=0.1,
            shift_xy=0.1,
            rotation=True,
            rotation_z=True,
            noise=0.1,
            threads=8,
            **kwargs,
            ):
        super(DataLoaderProcesser, self).__init__(**kwargs)
        self.size_change_z = size_change_z
        self.size_change_xy = size_change_xy
        self.shift_z = shift_z
        self.shift_xy = shift_xy
        self.rotation = rotation
        self.rotation_z = rotation_z
        self.noise = noise
        self.interpolator = InterpolatorCustom(
            shape=self.shape, threads=threads)

    def assign(self, input, grid):
        input = convert_video_to_occupancy(input)
        if self.augmentation:
            if self.noise > 0:
                input[:] += np.random.normal(0., self.noise, input.shape)
                input.clip(0, 1, out=input)
            size_z = np.random.normal(loc=1., scale=self.size_change_z)
            size_xy = np.random.normal(loc=1., scale=self.size_change_xy)
            shift_z = np.random.normal(scale=self.shift_z)
            shift_xy = np.random.normal(scale=self.shift_xy)
            angle_shift = np.pi * np.random.uniform()
            if self.rotation:
                theta = np.arccos(2 * np.random.random() - 1.)
                phi = 2 * np.pi * np.random.random()
                psi = 2 * np.pi * np.random.random()
            else:
                if self.rotation_z:
                    theta = np.pi / 2
                    phi = 0
                    psi = 2 * np.pi * np.random.random()
                else:
                    theta = 0
                    phi = 0
                    psi = 0
            self.interpolator.assign(input, 
                grid,
                size_z=size_z, 
                size_xy=size_xy,
                shift_z=shift_z,
                shift_x=np.cos(angle_shift)*shift_xy,
                shift_y=np.sin(angle_shift)*shift_xy,
                rotation_theta=theta,
                rotation_phi=phi, 
                rotation_psi=psi,
                )
            if random.getrandbits(1):
                grid[:] = np.flip(grid, 0)
            if random.getrandbits(1):
                grid[:] = np.flip(grid, 1)
            if random.getrandbits(1):
                grid[:] = np.flip(grid, 2)
            if not self.rotation and not self.rotation_z:
                r = np.random.randint(4)
                if r > 0:
                    grid[:] = np.rot90(grid, r, axes=(-2, -1))
        else:
            self.interpolator.assign(input, grid)