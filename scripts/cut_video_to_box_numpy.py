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
    read_video_into_numpy, crop_video, to_gray)

def cut_video_box(video, color_diff=20, add=10):
    video = crop_video(video, color_diff=color_diff, add=add)
    video = to_gray(video)
    return video

def save_video_cut_box(filename_in, filename_out, 
        color_diff=20, add=10):
    try:
        video = read_video_into_numpy(filename_in)
        video = cut_video_box(video, color_diff=color_diff, add=add)
        np.save(filename_out, video)
        return True
    except:
        print("Error:", filename_in, filename_out)
        return False

def save_video_cut_box_p(args):
    return save_video_cut_box(*args)

def run_cut_box(names, folder_in, folder_out, 
        color_diff=20, add=10, threads=1):
    def get_args():
        for name in names:
            filename_in = os.path.join(folder_in, name + ".mp4")
            filename_out = os.path.join(folder_out, name + ".npy")
            yield (filename_in, filename_out, color_diff, add)
    os.makedirs(folder_out, exist_ok=True)
    count_success = 0
    progress = tqdm(total=len(names))
    if threads == 1:
        for arg in get_args():
            count_success += save_video_cut_box(*arg)
            progress.update()
    else:
        pool = Pool(processes=threads)
        for res in pool.imap_unordered(save_video_cut_box_p, get_args()):
            count_success += res
            progress.update()
        pool.close()
        pool.join()
    progress.close()
    return count_success

def read_list(filename):
    with open(filename, "r") as file:
        return [f.strip().split(".")[0] for f in file.read().split("\n")]

def main(folder_in, folder_out, list_file=None,
        color_diff=20, add=10, threads=1, new=False):
    if list_file and os.path.isfile(list_file):
        names = read_list(list_file)
    else:
        names = [f.name.split(".")[0] for f in os.scandir(folder_in)
            if f.name.endswith(".mp4")]
    if not new and os.path.isdir(folder_out):
        names_done = [f.name.split(".")[0] for f in os.scandir(folder_out)]
        names_done_s = set(names_done)
        names = [n for n in names if n not in names_done_s]
    return run_cut_box(names, folder_in, folder_out,
        color_diff=color_diff, add=add, threads=threads)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input folder")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--list", help="path to file with list of files from input folder",
        default=None)
    parser.add_argument("--color_diff", help="threshold value for difference to define border",
        default=20, type=float)
    parser.add_argument("--add", help="number of pixels to add to size of the box at each side",
        default=10, type=int)
    parser.add_argument("--threads", help="number of processes for parallel conversion",
        default=1, type=int)
    parser.add_argument("--new", help="if set, old files will be rewritten",
        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    c = main(
        args.path, 
        args.out,
        list_file=args.list,
        color_diff=args.color_diff,
        add=args.add,
        threads=args.threads,
        new=args.new,
    )