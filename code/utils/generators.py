import cv2
from tensorflow.keras import Sequential
import numpy as np
from random import shuffle

def rotate(image, angle=45, scale=1.0):
    '''
    Rotate the image
    :param image: image to be processed
    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    :param scale: Isotropic scale factor.
    '''
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

def flip_vertical(image):
  #      """flip vertically"""
    return cv2.flip(image, 0)

def flip_horizontal(image):
    """flip horizontally"""
    return cv2.flip(image, 1)

def flip_both(image):
    """flip both vertically and horizontally"""
    return cv2.flip(image, -1)

def noisy(image, mean=0, sigma=0.1):
    """Gaussian-distributed additive noise."""
    row,col = image.shape
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    image = image + gauss
    return image

def random_transform(image):
    rn = random.randint(1, 5)
    print(rn)
    if rn == 1:
        return flip_vertical(image)
    elif rn == 2:
        return flip_horizontal(image)
    
    elif rn == 3:
        return flip_both(image)   
    elif rn == 4:
        return rotate(image)
    elif rn == 5:
        return noisy(image)

def video2tensor(mp4file, size=None):

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
                frame = crop_frame(frame, size)
                frames.append(frame)

            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
        else:
            break
    cap.release()

    tensor =  np.stack(frames, axis=0)
    
    return tensor



def crop_frame(frame, size=None):

    light_orange = (0, 80, 180)
    dark_orange = (80, 150, 255)
    
    positions = np.argwhere((frame[:, :, 0] < 100) & (frame[:, :, 1] > 100) & (frame[:, :, 1] < 180) & (frame[:, :, 2] > 200))
    y_min, x_min = np.amin(positions, axis=0)
    y_max, x_max = np.amax(positions, axis=0)

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    
    if size is None:

        size = ( int( (x_max - x_min) / 1), int( (y_max - y_min) / 1))

    mask = cv2.inRange(frame, light_orange, dark_orange)
    masked_positions = [[y,x] for x,y in zip(*np.where(mask))]
    masked_positions = np.array(masked_positions, dtype=np.int32)

    stencil = np.zeros(frame.shape).astype(frame.dtype)
    area = cv2.contourArea(masked_positions)
    cv2.fillPoly(stencil, pts=[masked_positions],color=(255, 255, 255))
    frame = cv2.bitwise_and(frame, stencil)
  
    mask = cv2.GaussianBlur(mask, (5, 5), 10)
    contour = cv2.bitwise_and(frame, frame, mask=mask)

    frame -= contour
    im = cv2.getRectSubPix(frame, size, center)
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    return gray

class BatchGenerator(Sequential):
    def __init__(self, mp4files, labels, size):
        
        self.files = list(zip(mp4files, labels))
        self.size = size

    def __data_generation(self, index):
        mp4file, label = self.files[index]
        tensor = video2tensor(mp4file, self.size) #(frames, width, heigth)
        tensor = tensor.astype('float32')
        tensor = np.expand_dims(tensor, axis=-1) #(frames, width, height, channels)
        return tensor, label
    
    
    def __len__(self):
        return len(self.files)

    

    def __getitem__(self, index):
        return self.__data_generation(index)

        