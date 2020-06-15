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
    mask_video_in_border, convert_video_to_occupancy,
    crop_volume, Interpolator,
)


"""
Script for converions of folder with mp4 video files
into folder with 3D numpy array grids of specified shape with 
volume occupancy obtained from video frames.
"""


def video_to_volume(
        video, 
        color_diff=20, 
        crop_threshold=0, 
        interpolator=Interpolator(),
        ):
    """
    Converts video array into volume.
    Args:
        video: np.array((frame_num, height, width, 3), np.uint8);
            numpy array with video;
        color_diff: float; 20;
            if difference of any pixel value with its average is larger
            than color_diff, it is considered to be on border;
        crop_threshold: float; 0;
            voxels with values <=threshold are considered as zeros;
        interpolator: Interpolator; Interpolator();
            object for grid volume interpolation.
    Returns:
        volume: np.array(interpolator.shape, np.float32);
            3D grid with occupancy.
    """
    video = crop_video(video, color_diff=color_diff)
    mask_video_in_border(video, color_diff=color_diff)
    volume = convert_video_to_occupancy(video)
    volume = crop_volume(volume, threshold=crop_threshold)
    return interpolator(volume)


def save_video_to_volume(
        filename_in, 
        filename_out,
        color_diff=20,
        crop_threshold=0,
        interpolator=Interpolator(),
        ):
    """
    Reads video from file, converts it and stores into npy file.
    Args:
        filename_in: str; 
            path to input mp4 file;
        filename_out: str;
            output npy file;
        color_diff: float; 20; (see video_to_volume);
        crop_threshold: float; 0; (see video_to_volume);
        interpolator: Interpolator; Interpolator(); (see video_to_volume).
    """
    video = read_video_into_numpy(filename_in)
    volume = video_to_volume(video, 
        color_diff=color_diff, 
        crop_threshold=crop_threshold, 
        interpolator=interpolator)
    np.save(filename_out, volume)
    return

def save_video_to_volume_p(args):
    return save_video_to_volume(*args)


def run_conversion(
        names, 
        folder_in, 
        folder_out, 
        color_diff=20, 
        crop_threshold=0,
        shape=(32, 32, 32),
        interpolate_method="linear",
        threads=1,
        ):
    """
    Converts list of video files.
    Args:
        names: list of str;
            list of filenames in folder_in (without .mp4 extension);
        folder_in: str;
            path to input folder with .mp4 video files;
        folder_out: str;
            folder to store npy files into;
        color_diff: float; 20; (see video_to_volume);
        crop_threshold: float; 0; (see video_to_volume);
        shape: tuple(3); (32, 32, 32);
            shape of output 3D grid;
        interpolate_method: str; "linear";
            method of interplation for scipy RegularGridInterpolation;
        threads: int; 1;
            number of processes.
    """
    interpolator = Interpolator(shape=shape, method=interpolate_method)
    def get_args():
        for name in names:
            filename_in = os.path.join(folder_in, name + ".mp4")
            filename_out = os.path.join(folder_out, name + ".npy")
            yield (filename_in, filename_out, 
                color_diff, crop_threshold, interpolator)
    os.makedirs(folder_out, exist_ok=True)
    progress = tqdm(total=len(names))
    if threads == 1:
        for arg in get_args():
            save_video_to_volume(*arg)
            progress.update()
    else:
        pool = Pool(processes=threads)
        for _ in pool.imap_unordered(save_video_to_volume_p, get_args()):
            progress.update()
        pool.close()
        pool.join()
    progress.close()
    return 


def read_list(filename):
    with open(filename, "r") as file:
        return [f.strip().split(".")[0] for f in file.read().split("\n")]


def main(
        folder_in, 
        folder_out,
        list_file=None,
        color_diff=20,
        crop_threshold=0,
        shape=(32, 32, 32),
        interpolate_method="linear",
        threads=1,
        ):
    if list_file and os.path.isfile(list_file):
        names = read_list(list_file)
    else:
        names = [f.name.split(".")[0] for f in os.scandir(folder_in)
            if f.name.endswith(".mp4")]
    return run_conversion(names, folder_in, folder_out, 
        color_diff=color_diff, crop_threshold=crop_threshold, shape=shape,
        interpolate_method=interpolate_method, threads=threads)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input folder")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--list", help="path to file with list of files from input folder",
        default=None)
    parser.add_argument("--color_diff", help="threshold value for difference to define border",
        default=20, type=float)
    parser.add_argument("--crop_threshold", help="threshold value for defining bounding box of volume",
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
    main(
        args.path, 
        args.out,
        list_file=args.list,
        color_diff=args.color_diff,
        crop_threshold=args.crop_threshold,
        shape=(args.shape_0, args.shape_12, args.shape_12),
        interpolate_method=args.interpolate_method,
        threads=args.threads,
    )