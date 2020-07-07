import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse

__folder_current = os.path.abspath(os.path.dirname(__file__))
__folder = os.path.join(__folder_current, "..")
__folder_code = os.path.join(__folder, "code")
sys.path.append(__folder_code)
from processing.convert import (
    convert_video_to_occupancy, crop_volume, Interpolator,)


def load_interpolated(filename, crop_threshold=0, interpolator=Interpolator()):
    volume = np.load(filename)
    volume = convert_video_to_occupancy(volume)
    volume = crop_volume(volume, threshold=crop_threshold)
    return interpolator(volume)

def load_interpolated_p(args):
    return load_interpolated(*args)

def main(folder, 
        crop_threshold=0, 
        shape=(32, 32, 32), 
        interpolate_method="linear",
        threads=1):
    names = [f.name for f in os.scandir(folder)]
    interpolator = Interpolator(shape=shape, method=interpolate_method)
    def get_args():
        for name in names:
            filename = os.path.join(folder, name)
            yield filename, crop_threshold, interpolator
    progress = tqdm(total=len(names))
    if threads == 1:
        for arg in get_args():
            load_interpolated(*arg)
            progress.update()
    else:
        pool = Pool(processes=threads)
        for res in pool.imap_unordered(load_interpolated_p, get_args()):
            progress.update()
        pool.close()
        pool.join()
    progress.close()
    return 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input folder")
    parser.add_argument("--crop_threshold", help="threshold value for defining bounding box",
        default=0, type=float)
    parser.add_argument("--shape_0", help="output grid shape along 0 axis",
        default=32, type=int)
    parser.add_argument("--shape_12", help="output grid shape along 1 and 2 axis",
        default=32, type=int)
    parser.add_argument("--interpolate_method", help="method for volume grid interpolation",
        default="linear")
    parser.add_argument("--threads", help="number of processes for parallel conversion",
        default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.path, 
        crop_threshold=args.crop_threshold,
        shape=(args.shape_0, args.shape_12, args.shape_12),
        interpolate_method=args.interpolate_method,
        threads=args.threads,
    )