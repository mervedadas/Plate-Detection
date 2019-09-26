import cv2
import numpy as np
def foreground(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    # define the Region of Interest (ROI)
    # it may vary for different images
    rectangle = (100, 400, 400,400)
    # apply the grabcut algorithm with appropriate
    # values as parameters, number of iterations = 3
    cv2.grabCut(img, mask, rectangle,  backgroundModel, foregroundModel,  3, cv2.GC_INIT_WITH_RECT)
    # In the new mask image, pixels will
    # be marked with four flags
    # four flags denote the background / foreground
    # mask is changed, all the 0 and 2 pixels
    # are converted to the background
    # mask is changed, all the 1 and 3 pixels
    # are now the part of the foreground
    # the return type is also mentioned,
    # this gives us the final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # The final mask is multiplied with
    # the input image to give the segmented image.
    img = img * mask2[:, :, np.newaxis]
    return img