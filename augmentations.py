import cv2
import random
"""input: grayscale 2D image
   otput: transformed 2D image"""
class Data_augmentation:
    def __init__(self):
        pass

    def rotate(self, image, angle=45, scale=1.0):
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
    
    
    def flip_vertical(self, image):
  #       """flip vertically"""
        return cv2.flip(image, 0)
    
    def flip_horizontal(self, image):
        """flip horizontally"""
        return cv2.flip(image, 1)
    
    def flip_both(self, image):
        """flip both vertically and horizontally"""
        return cv2.flip(image, -1)
    
    def noisy(self, image, mean=0, sigma=0.1):
        """Gaussian-distributed additive noise."""
        row,col = image.shape
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        image = image + gauss
        return image
    
    def random_transform(self, image):
        rn = random.randint(1, 5)
        print(rn)
        if rn == 1:
            return self.flip_vertical(image)
        elif rn == 2:
            return self.flip_horizontal(image)
        
        elif rn == 3:
            return self.flip_both(image)   
        elif rn == 4:
            return self.rotate(image)
        elif rn == 5:
            return self.noisy(image)
    
        
    
    
 
