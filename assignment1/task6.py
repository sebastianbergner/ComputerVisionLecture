import cv2
from matplotlib import pyplot as plt
import helpers

img = cv2.imread("resized_photo.jpg", cv2.IMREAD_COLOR)
helpers.showimage(img)
#gaussian
images = list()
images.append(img)
for i in range(4):
    images.append(cv2.pyrDown(images[i]))

helpers.showimages(images, (18,6))
#laplacian
laplacian_images = list()

for i in range(4):
    laplacian_images.append(cv2.subtract(images[i], cv2.pyrUp(images[i+1])))

helpers.showimages(laplacian_images, (18,6))
cv2.waitKey()