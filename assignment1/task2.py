import cv2 as cv

im = cv.imread('photo.jpg', cv.IMREAD_UNCHANGED)

resized_im = cv.resize(im, (448, 336), interpolation=cv.INTER_CUBIC)
cv.imwrite('resized_photo.jpg', resized_im)

cv.imshow('resized image', resized_im)
cv.waitKey(0)
cv.destroyAllWindows()