import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm

from video_processing import average_and_crop_video, video2tensor

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    videos = glob("../../res/data/test/*.mp4")

    video_arrays = []
    average_array = []
    names = []

    for v in tqdm(videos):

        video_arrays.append(np.expand_dims(video2tensor(v, length=20, size=(32, 32)), axis=0))
        average_array.append(np.expand_dims(average_and_crop_video(v, size=(32, 32))[1], axis=0))
        
        mp4video = v.split("/")[-1]
        names.append(mp4video)


    dataset = list(zip(video_arrays, average_array, names))
    dataset = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    torch.save(dataset, '../../res/data/test_set.pth')

