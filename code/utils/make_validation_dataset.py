import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm

from video_processing import average_and_crop_video, video2tensor

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    videos = glob("../../res/data/micro_vessels/*.mp4")

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

    nano_path = "../../res/data/micro_vessels/micro/"
    
    for i, v in enumerate(tqdm(df.index.values)):
        
        mp4video = nano_path + v
        
        video_tensor = np.expand_dims(video2tensor(mp4video, length=20, size=(32, 32)), axis=0)
        video_arrays.append(video_tensor)
        
        #one for a video tensor and another for flipped tensor
        average_array.append(np.expand_dims(average_and_crop_video(mp4video, size=(32, 32))[1], axis=0))
        
        labels.append(df.loc[v, 'label'])
        
        names.append(v)

    
    dataset = list(zip(video_arrays, average_array, labels, names))
    dataset = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    torch.save(dataset, '../../res/data/micro_validation_set.pth')