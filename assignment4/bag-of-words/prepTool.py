import os
import zipfile
import cv2 as cv
import tqdm
import numpy as np

zip_file = "/home/sebastian/uibk/ss2023/computerVision/assignments/assignment4/data/leafs.zip"

def get_filelist_from_zip(zip_file):
    filelist = []
    with zipfile.ZipFile(zip_file, 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if(info.file_size == 0):
                continue
            filelist.append(info.filename)
    return filelist

def import_images_from_zip(zip_file, filelist, shape=(256, 256), color=cv.IMREAD_GRAYSCALE):
    cvimgs = []
    with zipfile.ZipFile(zip_file, 'r') as zfile:
        for path_num in tqdm.trange(len(filelist)):
            img = cv.imdecode(np.frombuffer(zfile.read(filelist[path_num]), np.uint8), color)
            img = cv.resize(img, shape)
            img = cv.normalize(src=img, dst=img, norm_type=cv.NORM_MINMAX)
            cvimgs.append(img)
    return cvimgs

def augment_img(image):
    h, w = image.shape
    rot_mat = cv.getRotationMatrix2D((h/2, w/2), angle=np.random.randint(-180, 180), scale=1)
    new_im = cv.warpAffine(src=image, M=rot_mat, dsize=(w, h))
    return new_im

def extract_all_categories_from_path(filelist, categorie_path_pos=1):
    categorie_list = []
    for f in filelist:
        splited_path = f.split("/")
        categorie_name = splited_path[categorie_path_pos]
        categorie_list.append(categorie_name)
    return categorie_list, filelist

def write_to_fs(filenames, cvimgs):
    for i, cvimg in enumerate(cvimgs):
        cv.imwrite(filename=filenames[i], img=cvimg)

def limit_by_categories(categorielist, filelist, limit_categorielist):
    new_categorielist = []
    new_filelist = []
    for i in range(len(categorielist)):
        categorie = categorielist[i]
        f = filelist[i]
        if categorie in limit_categorielist:
            new_categorielist.append(categorie)
            new_filelist.append(f)
    return new_categorielist, new_filelist


filelist = get_filelist_from_zip(zip_file)
#filelist = list(filter(lambda x: x.__contains__('P4') , filelist)) 

cvimgs = import_images_from_zip(zip_file, filelist)
categories, _ = extract_all_categories_from_path(filelist)

categorielist, filelist = limit_by_categories(categories, filelist, list(set(categories)))

print(f'leng catlist {len(categorielist)} leng filelist {len(filelist)}')

imgs_per_cat = {}
new_imgs_per_cat = {}

for cat in categories:
    imgs_per_cat.update({cat:[]})
    new_imgs_per_cat.update({cat:[]})

#add imgs per category into dict
for i in tqdm.trange(len(cvimgs)):
    category = categorielist[i]
    fileli = filelist[i]
    img = cvimgs[i]
    imgs_per_cat[category].append((fileli, img))

#generate new imgs
for k in imgs_per_cat:
    for v0, v1 in imgs_per_cat[k]:
        new = augment_img(v1)
        new_imgs_per_cat[k].append(( v0.replace('.JPG', '_augmented7.JPG'),new))

saveloc = '/home/sebastian/uibk/ss2023/computerVision/assignments/assignment4/data/augmented/'

for k in new_imgs_per_cat:
    for loc, img in new_imgs_per_cat[k]:
        # print(f'writing to {saveloc+loc}')
        if os.path.exists(os.path.dirname(saveloc+loc)):
            cv.imwrite(saveloc+loc, img*255)
        else:
            os.makedirs(os.path.dirname(saveloc+loc))
            cv.imwrite(saveloc+loc, img*255)
