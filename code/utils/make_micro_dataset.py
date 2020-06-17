import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm

from video_processing import average_and_crop_video, video2tensor, video2tensor_with_rotation

import torch
from torch.utils.data import DataLoader
from random import shuffle, randint

THRESHOLD = 0.75
TRAIN_SIZE = 0.8

if __name__ == "__main__":
    
    df = pd.read_csv("../../res/data/train_metadata.csv")
    df.set_index('filename', inplace=True)
    df = df.loc[df['micro'] == True]
    df['label'] = df['crowd_score'].apply(lambda x: 1 if x >= THRESHOLD else 0)
    print(df.head())
    print(df['num_frames'].describe())

    video_arrays = []
    average_array = []
    labels = []
    names = []

    nano_path = "../../res/data/micro/"
    
    for i, v in enumerate(tqdm(df.index.values)):
        
        mp4video = nano_path + v
        
        video_tensor = np.expand_dims(video2tensor(mp4video, length=20, size=(32, 32)), axis=0)
        average_scan = np.expand_dims(average_and_crop_video(mp4video, size=(32, 32))[1], axis=0)
        average_array.append(average_scan)
        video_arrays.append(video_tensor)
        labels.append(df.loc[v, 'label'])
        names.append(v)

        
        if df.loc[v, 'label']:

            flipped_video_tensor = np.flip(video_tensor, axis=-1)
            video_arrays.append(flipped_video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)

            video_tensor = np.expand_dims(video2tensor_with_rotation(mp4video, length=20, size=(32, 32), method=1), axis=0)
            video_arrays.append(video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)

            video_tensor = np.expand_dims(video2tensor_with_rotation(mp4video, length=20, size=(32, 32), method=2), axis=0)
            video_arrays.append(video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)

            video_tensor = np.expand_dims(video2tensor_with_rotation(mp4video, length=20, size=(32, 32), method=3), axis=0)
            video_arrays.append(video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)

            video_tensor = np.expand_dims(video2tensor_with_rotation(mp4video, length=20, size=(32, 32), method=4), axis=0)
            video_arrays.append(video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)

        else:
            method = randint(1, 4)
            video_tensor = np.expand_dims(video2tensor_with_rotation(mp4video, length=20, size=(32, 32), method=method), axis=0)
            video_arrays.append(video_tensor)
            average_array.append(average_scan)
            labels.append(df.loc[v, 'label'])
            names.append(v)


    dataset = list(zip(video_arrays, average_array, labels, names))
    shuffle(dataset)

    train = dataset[:int(len(dataset) * TRAIN_SIZE)]
    test = dataset[int(len(dataset) * TRAIN_SIZE):]
    
    train = DataLoader(train, batch_size=128, shuffle=True, pin_memory=True)
    torch.save(train, '../../res/data/micro_train_set_augmented.pth')

    test = DataLoader(test, batch_size=128, shuffle=True, pin_memory=True)
    torch.save(test, '../../res/data/micro_test_set_augmented.pth')

