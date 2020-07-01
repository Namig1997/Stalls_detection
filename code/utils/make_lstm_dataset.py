from video_processing import video2features
import numpy as np
import pandas as pd
from random import choice
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


def generate_batches(groups, min_frames=0, max_frames=99999, n_threads=1):
    path = "../../res/data/micro/"

    X = []
    y = []
    
    frame_groups = [g for g in groups if g > min_frames]
    frame_groups = [g for g in frame_groups if g < max_frames]
    print(frame_groups)
    for g in tqdm(frame_groups):
        videos = df.iloc[groups[g], 0].values
        labels = []
        
        arrs = []
        for v in videos:
            videofile= path + v
            try:
                arr = video2features(videofile, num_frames=g, size=(64, 64))
                arr = arr.T
                label = df.loc[df['filename'] == v]['labels'].values[0]
                labels.append(label)
            except ValueError:
                continue
            arrs.append(arr.reshape((1, arr.shape[0], -1))/255.0)
            

        X.append(np.vstack(arrs))
        y.append(labels)
    print(len(X))
    print(len(y))
    return X, y

def generate_test_set(filenames):

    path = "../../res/data/test/"

    X_sub = []
    for f in tqdm(filenames):
        videofile = path + f
        
        arr = video2features(videofile, num_frames=None, size=(64, 64))
        arr = arr.T
        arr = arr.reshape((arr.shape[0], -1))/255.0
        #arr = np.vstack(arr)
        X_sub.append(arr)

    return X_sub

if __name__ == "__main__":
    
    df = pd.read_csv("../../res/data/train_metadata.csv")
    df = df.loc[df['micro'] == True]
    df['labels'] = df['crowd_score'].apply(lambda x: 1 if x > 0.75 else 0)

    groups = df.groupby('num_frames').indices

    #X, y = generate_batches(groups)
    #X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                    test_size=0.2)

    #np.savez('../../res/data/lstm_set', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    submissons = pd.read_csv('../../res/data/test_metadata.csv')
    print(submissons.head())
    subfiles = submissons['filename'].values
    X_sub = generate_test_set(subfiles)
    print(X_sub[0].shape)
    print(len(X_sub))
    np.savez('../../res/data/submission_lstm_set', X_sub=X_sub, filenames=subfiles)
