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
    read_video_into_numpy, crop_video, 
    mask_video_in_border, crop_volume, to_gray,)


def cut_video(video, color_diff=20):
    video = crop_video(video, color_diff=color_diff)
    mask_video_in_border(video, color_diff=color_diff)
    video = to_gray(video)
    video = crop_volume(video)
    return video


def save_video_cut(filename_in, filename_out, color_diff=20):
    video = read_video_into_numpy(filename_in)
    video = cut_video(video, color_diff=color_diff)
    np.save(filename_out, video)
    return 

def save_video_cut_p(args):
    return save_video_cut(*args)


def run_cut(names, folder_in, folder_out, color_diff=20, threads=1):
    def get_args():
        for name in names:
            filename_in = os.path.join(folder_in, name + ".mp4")
            filename_out = os.path.join(folder_out, name + ".npy")
            yield (filename_in, filename_out, color_diff)
    os.makedirs(folder_out, exist_ok=True)
    progress = tqdm(total=len(names))
    if threads == 1:
        for arg in get_args():
            save_video_cut(*arg)
            progress.update()
    else:
        pool = Pool(processes=threads)
        for _ in pool.imap_unordered(save_video_cut_p, get_args()):
            progress.update()
        pool.close()
        pool.join()
    progress.close()
    return

def read_list(filename):
    with open(filename, "r") as file:
        return [f.strip().split(".")[0] for f in file.read().split("\n")]

def main(folder_in, folder_out, list_file=None, color_diff=20, threads=1):
    if list_file and os.path.isfile(list_file):
        names = read_list(list_file)
    else:
        names = [f.name.split(".")[0] for f in os.scandir(folder_in)
            if f.name.endswith(".mp4")]
    return run_cut(names, folder_in, folder_out, 
        color_diff=color_diff, threads=threads)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input folder")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--list", help="path to file with list of files from input folder",
        default=None)
    parser.add_argument("--color_diff", help="threshold value for difference to define border",
        default=20, type=float)
    parser.add_argument("--threads", help="number of processes for parallel conversion",
        default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.path, 
        args.out,
        list_file=args.list,
        color_diff=args.color_diff,
        threads=args.threads,
    )