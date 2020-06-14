import cv2
import numpy as np
from multiprocessing import Pool


class FrameCropper(object):

    def __init__(self, n_cpus=1):
        self.mp = Pool(n_cpus)

    def crop_frame(self, frame, size):
    
        positions = np.argwhere(frame[:, :, -1] > 150)
        x_min, y_min = np.amin(positions, axis=0)
        x_max, y_max = np.amax(positions, axis=0)

        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        if size is None:
            buf = 0
            size = ( int(buf + (x_max - x_min) / 2 ), int(buf + (y_max - y_min) / 2))
            print(size)
            print(center)


        return cv2.getRectSubPix(frame, size, center)
    
    
    def __call__(self, mp4file):
        cap = cv2.VideoCapture(mp4file)
        frames = []
        while(cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = cap.read()
          if ret:
            frames.append(crop_frame(frame))

        # When everything done, release the video capture object
        cap.release()

        
