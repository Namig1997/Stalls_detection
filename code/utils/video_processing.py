import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
import plotly.express as px
from plotly.offline import plot

def apply_invert(frame):
    return cv2.bitwise_not(frame)


def apply_rotation(frame, method=1):
    if method==1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if method==2:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if method==3:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if method==4:
        return cv2.flip(frame, -1)
    


def video2tensor_with_rotation(mp4file, length=20, size=None, method=1):

    cap = cv2.VideoCapture(mp4file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    
    frames = []
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if size is not None:
                frames.append(crop_frame(frame, size)[1])
            else:
                frames.append(frame)
        else:
            break
    cap.release()

    starting_frame = len(frames) // 2 - length // 2
    ending_frame = len(frames) // 2 + length // 2
    frames = frames[starting_frame:ending_frame]
    frames = [apply_rotation(frame, method=method) for frame in frames]
    tensor =  np.stack(frames, axis=-1)
    
    return tensor


def video2tensor(mp4file, length=20, size=None):

    cap = cv2.VideoCapture(mp4file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    
    frames = []
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if size is not None:
                frame = crop_frame(frame, size)[1]
                frames.append(frame)

            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
        else:
            break
    cap.release()

    starting_frame = len(frames) // 2 - length // 2
    starting_frame = max(0, starting_frame)
    ending_frame = len(frames) // 2 + length // 2
    frames = frames[starting_frame:ending_frame]
    tensor =  np.stack(frames, axis=-1)
    
    return tensor

def video2features(mp4file, num_frames=None, size=None):

    cap = cv2.VideoCapture(mp4file)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (length != num_frames) and (num_frames is not None):
        raise ValueError('Num of frames is not equal to length of the video')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    
    frames = []
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if size is not None:
                frame = crop_frame(frame, size)[1]
                frames.append(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
        else:
            break
    cap.release()

    starting_frame = len(frames) // 2 - length // 2
    starting_frame = max(0, starting_frame)
    ending_frame = len(frames) // 2 + length // 2
    frames = frames[starting_frame:ending_frame]
    tensor =  np.stack(frames, axis=-1)
    
    return tensor
    

def crop_frame(frame, size=None):

    light_orange = (0, 80, 180)
    dark_orange = (80, 150, 255)
    
    positions = np.argwhere((frame[:, :, 0] < 100) & (frame[:, :, 1] > 100) & (frame[:, :, 1] < 180) & (frame[:, :, 2] > 200))
    y_min, x_min = np.amin(positions, axis=0)
    y_max, x_max = np.amax(positions, axis=0)

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    
    if size is None:
        buf = 0
        size = ( int(buf + (x_max - x_min) / 1), int(buf + (y_max - y_min) / 1))

    mask = cv2.inRange(frame, light_orange, dark_orange)
    masked_positions = [[y,x] for x,y in zip(*np.where(mask))]
    masked_positions = np.array(masked_positions, dtype=np.int32)

    stencil = np.zeros(frame.shape).astype(frame.dtype)
    area = cv2.contourArea(masked_positions)
    cv2.fillPoly(stencil, pts=[masked_positions],color=(255, 255, 255))
    frame = cv2.bitwise_and(frame, stencil)
  
    
    #mask = cv2.inRange(frame, light_orange, dark_orange)
    mask = cv2.GaussianBlur(mask, (5, 5), 10)
    contour = cv2.bitwise_and(frame, frame, mask=mask)
    #print(contour.shape)
    frame -= contour
    im = cv2.getRectSubPix(frame, size, center)
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #ret, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    #gray = cv2.bilateralFilter(gray, 15, 75, 75)
    #gray = cv2.adaptiveThreshold(gray,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,3)

    return im, gray, size, area, np.sum(gray)

def average_and_crop_video(mp4file, size=None):
    
    cap = cv2.VideoCapture(mp4file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3
    
    avg = np.zeros((heigth, width, channels))
    # Read until video is completed
    frames = 0
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            avg = np.add(avg, frame)
            frames += 1
        else:
            break
  
    cap.release()
    avg = avg // frames
    avg = avg.astype(np.uint8)
    
    if size is not None:
        return crop_frame(avg, size=size)
        
    return crop_frame(avg)



if __name__ == "__main__":

    if False:
        #193536.mp4 surface area 0 
        #104360.mp4
        im, gr, size, area, _ = average_and_crop_video('../../res/data/nano_vessels/nano/193536.mp4', size=(32, 32))
        print(gr.shape)
        print("size: ", size)
        print("surface area: ", area)
        cv2.imshow('frame', im)
        cv2.imshow('gray', gr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    if False:
        df = pd.read_csv("train_metadata.csv")
        df.set_index('filename', inplace=True)
        df['dim1'] = None
        df['dim2'] = None
        df['area'] = None
        df['gray'] = None

        videos = glob("nano_vessels/nano/*.mp4")
        sizes = []
        areas = []

        for v in tqdm(videos):

            _, size, area, gray = average_and_crop_video(v)
            sizes.append(size)
            areas.append(area)

            name = v.split("/")[-1]
            df.loc[name, 'dim1'] = size[0]
            df.loc[name, 'dim2'] = size[1]
            df.loc[name, 'area'] = area
            df.loc[name, 'gray'] = gray


        df.to_csv("train_metadata_v01.csv")



    
