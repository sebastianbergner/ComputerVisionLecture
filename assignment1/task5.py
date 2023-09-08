import cv2
import helpers

img = cv2.imread("resized_photo.jpg", cv2.IMREAD_COLOR)
helpers.showimage(img)

images = list()
images.append(img)
for i in range(3):
    images.append(cv2.pyrDown(images[i]))

#convert color from bgr to rgb
for i in range(len(images)):
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

helpers.showimages(images, (18,6))
