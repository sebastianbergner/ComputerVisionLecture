import cv2 as cv
import numpy as np
import helpers

"""
In this equation, 
lambda represents the wavelength of the sinusoidal factor, 
theta represents the orientation of the normal to the parallel stripes of a Gabor function, 
psi is the phase offset, 
sigma is the sigma/standard deviation of the Gaussian envelope and 
gamma is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.
"""

im = cv.imread('resized_photo.jpg', cv.IMREAD_GRAYSCALE)

def create_gaborfilter(num_filters = 4, kernel_size=35, sigma=1.5, lambd=3.5, gamma=0.5, psi=0.1):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()
        filters.append(kern)
    return filters

def apply_filter(img, filters):
    newimage = np.zeros_like(img)
    for kern in filters:
        image_filter = cv.filter2D(img, -1, kern)
        np.maximum(newimage, image_filter, newimage)
    return newimage

gfilters = create_gaborfilter()
image_g = apply_filter(im, gfilters)


helpers.showimages(gfilters, (18, 6), txt='num_filters = 4, kernel_size=35, sigma=1.5, lambd=3.5, gamma=0.5, psi=0.1')
helpers.showimage(image_g)
