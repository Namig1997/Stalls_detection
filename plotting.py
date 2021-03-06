import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
import plotly.express as px
from plotly.offline import plot

if __name__ == "__main__":
    
    df = pd.read_csv("train_metadata_v01.csv", index_col=0)
    df.dropna(inplace=True)
    print(df.head())
    df = df.loc[df['tier1'] == True]
    df['label'] = df['crowd_score'].apply(lambda x: 1 if x > 0.5 else 0)
    df['normalized_gray'] = df['gray'] / (df['area'] + 1)

    print(df['area'].sort_values())
    print(df.loc[df['crowd_score'] > 0.75])
    if False:
        fig = px.histogram(df, x='normalized_gray', color='label', barmode='overlay', histnorm='percent')
        plot(fig, filename='normalized_gray_hist.html')

        fig = px.histogram(df, x='gray', color='label', barmode='overlay', histnorm='percent')
        plot(fig, filename='gray_hist.html')

        fig = px.histogram(df, x='area', color='label', barmode='overlay', histnorm='percent')
        plot(fig, filename='areas_hist.html')