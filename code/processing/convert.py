import os
import numpy as np
import cv2


COLOR_DIFF = 20


def read_video_into_numpy(filename):
    """
    Reads video from file and stores into single numpy array.
    Args:
        filename: str; 
            path to input file.
    Returns:
        frames: np.array((frame_num, height, width, 3), np.uint8);
            numpy array with set of image frames.
    """
    video = cv2.VideoCapture(filename)
    frame_count  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((frame_count, frame_height, frame_width, 3), np.uint8)
    for i in range(frame_count):
        _, frame = video.read()
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frames


def to_gray(video):
    if len(video.shape) == 4:
        video_gray = np.empty(video.shape[:3], np.uint8)
        for i in range(len(video)):
            video_gray[i] = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        return video_gray
    else:
        return cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)


def get_border(image, color_diff=COLOR_DIFF):
    """
    Returns indexes of pixels of specified vessel border contour.
    As it is supposed to be orange, here we just get pixels that
    are not of shade of gray color.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        color_diff: float; 20;
            if difference of any pixel value with its average is larger
            than color_diff, it is considered to be on border.
    Returns:
        border_indexes: list [border_indexes_x, border_indexes_y];
            indexes of border pixels.
    """
    image_mean = image.mean(axis=-1)
    diff = np.max(np.abs([
        image[:, :, 0]-image_mean,
        image[:, :, 1]-image_mean,
        image[:, :, 2]-image_mean]), axis=0)
    border_indexes = np.where(diff > color_diff)
    return border_indexes


def get_bounding_box(image, color_diff=COLOR_DIFF):
    """
    Gets bouding box around the region defined by contour.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        color_diff: float; 20;
            if difference of any pixel value with its average is larger
            than color_diff, it is considered to be on border.
    Returns:
        rect: np.array((4), np.int32);
            bounding box [min_x, min_y, max_x, max_y].
    """
    border_indexes = get_border(image, color_diff=color_diff)
    rect = np.zeros((4), np.int32)
    rect[0] = np.min(border_indexes[0])
    rect[1] = np.min(border_indexes[1])
    rect[2] = np.max(border_indexes[0])+1
    rect[3] = np.max(border_indexes[1])+1
    return rect


def crop_image_rect(image, rect, add=0):
    """
    Crops image inside defined rectangle.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        rect: np.array((4), np.int32); None;
            bounding box [min_x, min_y, max_x, max_y] of pixel indexes to leave;
        add: int; 0;
            value to add to rectangle at each side.
    Returns:
        image: np.array((rect_size_x, rect_size_y, 3), np.uint8);
            cropped image.
    """
    if rect is None:
        rect = np.array([0, 0, image.shape[0], image.shape[1]])
    else:
        rect = rect.copy()
    rect[0] = max(rect[0]-add, 0)
    rect[1] = max(rect[1]-add, 0)
    rect[2] = min(rect[2]+add, image.shape[0])
    rect[3] = min(rect[3]+add, image.shape[1])
    return image[rect[0]:rect[2], rect[1]:rect[3]]


def crop_image(image, color_diff=COLOR_DIFF, add=0):
    """
    Crops image inside bounding box defined by contour.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        color_diff: float; 20; (see get_border);
        add: int; 0; (see crop_image_rect).
    Returns:
        image: np.array((height_cropped, width_cropped, 3), np.uint8);
            cropped image.
    """
    rect = get_bounding_box(image, color_diff=color_diff)
    return crop_image_rect(image, rect, add=add)


def crop_video(video, color_diff=COLOR_DIFF, add=0, all_rect=False):
    """
    Crops image inside bounding box defined by contour.
    Args:
        video: np.array((frame_num, height, width, 3), np.uint8);
            input video;
        color_diff: float; 20; (see get_border);
        add: int; 0; (see crop_image_rect);
        all_rect: bool; False;
            if True, rectangles for each frame are retrieved and 
            then the maximum is used; else rectangle of the first one is used.
    Returns:
        video: np.array((frame_num, height_cropped, width_cropped, 3), np.uint8);
            cropped video.
    """
    if all_rect:
        rect_list = [get_bounding_box(image, color_diff=color_diff) for image in video]
        rect_list = np.array(rect_list)
        # for video, rectangle is defined as bounding for all frames
        rect = np.zeros((4), np.int32)
        rect[0] = rect_list[:, 0].max()
        rect[1] = rect_list[:, 1].max()
        rect[2] = rect_list[:, 2].min()
        rect[3] = rect_list[:, 3].min()
    else:
        rect = get_bounding_box(video[0], color_diff=color_diff)
    image_0 = crop_image_rect(video[0], rect, add=add)
    video_cropped = np.empty((len(video), image_0.shape[0], image_0.shape[1], 3), 
        dtype=video.dtype)
    video_cropped[0] = image_0
    for i in range(1, len(video)):
        video_cropped[i] = crop_image_rect(video[i], rect, add=add)
    return video_cropped


def get_mask_without_border(image, color_diff=COLOR_DIFF):
    """
    Gets boolean mask of image. True corresponds to region inside borders.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        color_diff: float; 20; (see get_border).
    Returns:
        mask: np.array((height, width), np.bool);
            boolean mask of region inside border contours.
    """
    border_indexes = get_border(image, color_diff=color_diff)
    # get indexes of min and max border pixels in specified row
    indexes_0 = []
    for i in range(image.shape[0]):
        ind = np.where(border_indexes[0] == i)[0]
        if len(ind) == 0:
            indexes_0.append([])
        else:
            indexes_0.append([
                np.min(border_indexes[1][ind]),
                np.max(border_indexes[1][ind]),
            ])
    # get indexes of min and max border pixels in specified column
    indexes_1 = []
    for i in range(image.shape[1]):
        ind = np.where(border_indexes[1] == i)[0]
        if len(ind) == 0:
            indexes_1.append([])
        else:
            indexes_1.append([
                np.min(border_indexes[0][ind]),
                np.max(border_indexes[0][ind]),
            ])
    mask = np.ones((image.shape[0], image.shape[1]), np.bool)
    mask[border_indexes] = 0
    # iterage over pixels to check whether it lays inside borders
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if [i, j] in border_indexes:
                continue
            if len(indexes_0[i]) == 0 or len(indexes_1[j]) == 0 or \
                j <= indexes_0[i][0] or j >= indexes_0[i][1] or \
                i <= indexes_1[j][0] or i >= indexes_1[j][1]:
                mask[i, j] = 0
    return mask


def mask_image(image, mask):
    """
    Multiplies image by boolean mask leaving nonzero only 
    values inside border region.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        mask: np.array((height, width), np.bool);
            boolean mask of region inside border contours.
    Returns:
        image: np.array((height, width, 3), np.uint8);
            masked image.
    """
    image[:, :, 0] *= mask
    image[:, :, 1] *= mask
    image[:, :, 2] *= mask
    return image


def mask_image_in_border(image, color_diff=COLOR_DIFF):
    """
    Masks image inside region specified by border contours.
    Args:
        image: np.array((height, width, 3), np.uint8);
            input image;
        mask: np.array((height, width), np.bool);
            boolean mask of region inside border contours.
    Returns:
        image: np.array((height, width, 3), np.uint8);
            masked image.
    """
    mask = get_mask_without_border(image, color_diff=color_diff)
    return mask_image(image, mask)


def mask_video_in_border(video, color_diff=COLOR_DIFF):
    """
    Masks video inside region specified by border contours.
    Args:
        video: np.array((frame_num, height, width, 3), np.uint8);
            input video;
        color_diff: float; 20; (see get_border).
    Returns:
        video: np.array((frame_num, height, width, 3), np.uint8);
            masked video.
    """
    mask = get_mask_without_border(video[0], color_diff=color_diff)
    for i in range(len(video)):
        mask_image(video[i], mask)
    return video


def convert_image_to_occupancy(image):
    if len(image.shape) == 3:
        return np.array(image.mean(axis=-1) / 255, np.float32)
    else:
        return np.array(image / 255, np.float32)

def convert_video_to_occupancy(video):
    if len(video.shape) == 4:
        return np.array(video.mean(axis=-1) / 255, np.float32)
    else:
        return np.array(video / 255, np.float32)


def get_volume_bouding_box(volume, threshold=0):
    """
    Gets bounding box for region of the volume with 
    voxels > threshold.
    Args:
        volume: np.array((z, x, y), np.float32);
            3d array of voxels;
        threshold: float; 0;
            voxels with values <=threshold are considered as zeros.
    Returns:
        box: np.array((6), np.int32);
            bounding box [min_z, min_x, min_y, max_z, max_x, max_y].
    """
    mask_ind = np.where(volume > threshold)
    box = np.zeros((6), np.int32)
    box[0] = np.min(mask_ind[0])
    box[1] = np.min(mask_ind[1])
    box[2] = np.min(mask_ind[2])
    box[3] = np.max(mask_ind[0])+1
    box[4] = np.max(mask_ind[1])+1
    box[5] = np.max(mask_ind[2])+1
    return box

def crop_volume(volume, threshold=0):
    """
    Crops volume for nonzero region.
    Args:
        volume: np.array((z, x, y,), np.float32);
            3d array of voxels;
        threshold: float; 0; (see get_volume_bounding_box).
    Returns:
        volume: np.array((box_z, box_x, box_y), np.float32);
            cropped volume.
    """
    box = get_volume_bouding_box(volume, threshold=threshold)
    return volume[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
        