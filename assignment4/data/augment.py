import numpy as np
import cv2 as cv
import tqdm
import os
import sys

def random_transformation(img):
    rand = np.random.rand()
    
    if rand <= np.random.rand():
        # print('mirror/flip')
        dir = np.array([-1, 0, 1])
        np.random.shuffle(dir)
        img = cv.flip(img, dir[0])
    if rand <= np.random.rand():
        # print('rotate')
        h, w = img.shape[:-1]
        dir = np.array([-180, -90, 90, 180])
        np.random.shuffle(dir)
        rot_mat = cv.getRotationMatrix2D((h/2, w/2), angle=dir[0], scale=1)
        img = cv.warpAffine(src=img, M=rot_mat, dsize=(h, w))
    if rand <= np.random.rand():
        # print('darken/brighten')
        img = cv.convertScaleAbs(img, beta=np.random.randint(-20, 20))
    if rand <= np.random.rand():
        # print('contrast')
        img = cv.convertScaleAbs(img, alpha=np.random.randint(50,200)/100.0)
    if rand <= np.random.rand():
        # print('saturation')
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv.split(img)
        s = s*np.random.randint(80,200)/100.0
        s = np.clip(s,0,255)
        img = cv.merge([h,s,v])
        img = cv.cvtColor(img.astype("uint8"), cv.COLOR_HSV2BGR)
    else:
        # print('scale and crop')
        h, w = img.shape[:-1]
        new_size = np.random.randint(h,int(h*1.1))
        img = cv.resize(img, (new_size, new_size))
        img = img[0:h, 0:w]
    return img


if __name__ == "__main__":
    top_up_to = 1000
    directory = '/home/sebastian/uibk/ss2023/computerVision/assignments/assignment4/data/A Database of Leaf Images Practice towards Plant Conservation with Plant Pathology/'
    for filename in os.listdir(directory):
        category = filename
        if not (category.__contains__('Basil') or category.__contains__('Bael')):
            continue
        for filename in os.listdir(os.path.join(directory, category)):
            healthy_diseased = filename
            num_files = len([entry for entry in os.listdir(os.path.join(directory, category, healthy_diseased)) if os.path.isfile(os.path.join(os.path.join(directory, category, healthy_diseased), entry))])
            print(num_files)
            filelist = os.listdir(os.path.join(directory, category, healthy_diseased))
            for i in tqdm.trange(top_up_to-num_files):
                #select random img
                img_name = os.path.join(directory, category, healthy_diseased, filelist[np.random.randint(num_files-1)])
                
                img = cv.imread(img_name, cv.IMREAD_COLOR)
                img = cv.resize(img, (3000,3000))
                # cv.imshow('frame', (cv.resize(img, (720,576))))
                # while(1):
                #     if cv.waitKey(1) & 0xFF == ord('q'):
                #         break
                #apply random change
                img = random_transformation(img)
                # cv.imshow('frame', (cv.resize(img, (720,576))))
                # while(1):
                #     if cv.waitKey(1) & 0xFF == ord('q'):
                #         break
                #save
                cv.imwrite(img_name.replace('.JPG', 'a'+str(i)+'.JPG'), img)
    # random_transformation('s')