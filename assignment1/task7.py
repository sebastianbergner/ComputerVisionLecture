import helpers
import cv2 as cv
import numpy as np

im = cv.imread('photo.jpg', cv.IMREAD_GRAYSCALE)

m = cv.getOptimalDFTSize(im.shape[0])
n = cv.getOptimalDFTSize(im.shape[1])

#add border to make img optimal for discrete fourier
im_border = cv.copyMakeBorder(im, 0, m-im.shape[0], 0, n-im.shape[1], cv.BORDER_CONSTANT, None, 0)

#show image with the added border
helpers.showimage(im_border)

#generate array to store the generated img
planes = np.array([np.copy(im_border), np.zeros(im_border.shape, np.float32)])

complexI = cv.merge(planes)
complexI = cv.dft(complexI)

cv.split(complexI, planes)

magI = cv.magnitude(planes[0], planes[1])
magI += 1.0
magI = cv.log(magI)
magI = magI[0:(magI.shape[0] & -2), 0:(magI.shape[1] & -2)]
helpers.swap_quadrants(magI)
magI = cv.normalize(magI, None, 0, 1, cv.NORM_MINMAX)
helpers.showimage(magI)

phase = cv.phase(planes[0], planes[1])
phase += 1.0
phase = cv.log(phase)
phase = phase[0:(phase.shape[0] & -2), 0:(phase.shape[1] & -2)]
helpers.swap_quadrants(phase)
phase = cv.normalize(phase, None, 0, 1, cv.NORM_MINMAX)
helpers.showimage(phase)
