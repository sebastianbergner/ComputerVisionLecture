import cv2
import numpy as np
from matplotlib import pyplot as plt


def showimage(myimage, figsize=[10,10]):
    if (myimage.ndim>2):
        myimage = myimage[:,:,::-1]
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(myimage, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

def showimages(images, figuresize, cmap='gray', txt=''):
    _, ax = plt.subplots(1, len(images), figsize=figuresize, dpi=120)
    for i, image in enumerate(images):
        ax[i].imshow(image, cmap=cmap)
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=14)
    plt.show()

def swap_quadrants(img):
    cx = int(img.shape[0]/2)
    cy = int(img.shape[1]/2)
    q0 = img[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = img[cx:cx+cx, 0:cy]     # Top-Right
    q2 = img[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = img[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    img[0:cx, 0:cy] = q3
    img[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    img[cx:cx + cx, 0:cy] = q2
    img[0:cx, cy:cy + cy] = tmp