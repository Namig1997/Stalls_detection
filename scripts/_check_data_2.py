import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from data import (
    DataLoaderReader, DataLoaderProcesser,
)


def main(folder, shape=(32, 32, 32), batch_size=8):
    names = [f.name.split(".")[0] for f in os.scandir(folder)]
    dataloader = DataLoaderProcesser(
        folder=folder,
        names=names,
        shape=shape,
        threads=8,
    )
    # dataloader2 = DataLoaderProcesser(
    #     folder=folder,
    #     names=names,
    #     shape=shape,
    #     tt=2,
    # )
    batch_number = dataloader.get_batch_number(batch_size)
    batch_generator = dataloader.batch_generator(batch_size)
    # batch_generator2 = dataloader2.batch_generator(batch_size)

    batch_number = 100

    for _ in tqdm(range(batch_number)):
        g = next(batch_generator)
        # g2 = next(batch_generator2)
        # print(g.shape, g.min(), g.max(), g.mean())
        # print(g2.shape, g2.min(), g2.max(), g2.mean())
        # g3 = g2 - g
        # print(g3.min(), g3.max(), g3.mean())
        # for i in range(dataloader.shape[0]):
        #     for j in range(dataloader.shape[1]):
        #         for k in range(dataloader.shape[2]):
        #             # if np.abs(g3[0, i, j, k]) > 0.3:
        #             print(i, j, k, g[0, i, j, k], g2[0, i, j, k], g3[0, i, j, k])

    return


if __name__ == "__main__":
    main("./res/data/custom_test_pstalled_cut", (32, 32, 32), 64)
    main("./res/data/custom_test_pstalled_cut", (48, 48, 48), 64)
    main("./res/data/custom_test_pstalled_cut", (64, 64, 64), 64)
    