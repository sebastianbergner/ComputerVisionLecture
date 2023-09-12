import zipfile
import numpy as np
import cv2 as cv
import tqdm


def get_filelist_from_zip(zip_file):
    filelist = []
    with zipfile.ZipFile(zip_file, 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if(info.file_size == 0):
                continue
            filelist.append(info.filename)
    # filelist = [k for k in filelist if 'healthy' in k]
    return filelist


def extract_all_categories_from_path(filelist, categorie_path_pos=1):
    categorie_list = []
    for f in filelist:
        splited_path = f.split("/")
        categorie_name = splited_path[categorie_path_pos]
        categorie_list.append(categorie_name)
    return categorie_list, filelist

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

def import_images_from_zip(zip_file, filelist, color=cv.IMREAD_GRAYSCALE):
    cvimgs = []
    with zipfile.ZipFile(zip_file, 'r') as zfile:
        for path_num in tqdm.trange(len(filelist)):
            img = cv.imdecode(np.frombuffer(zfile.read(filelist[path_num]), np.uint8), color)
            img = cv.resize(img, (256,256))
            img = cv.GaussianBlur(img,(7,7),0)
            img = np.uint8(np.uint8((np.double(img)-np.double(np.min(np.min(img))))*(255/np.double(np.max(np.max(img))-np.min(np.min(img))))))
            cvimgs.append(img)
    return cvimgs