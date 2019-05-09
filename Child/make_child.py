import cv2
import glob
import os
import shutil

root = '/media/yj/hanson/face-recognition/chip_img_origin_top'
save = '/media/yj/hanson/face-recognition/chip_img_origin'
for path in glob.glob(os.path.join(root, '*.jpg')):
    id_folder = os.path.join(save, path.split('/')[-1].split('_')[0])
    if not os.path.exists(id_folder):
        os.makedirs(id_folder)
    shutil.copyfile(path, os.path.join(id_folder, path.split('/')[-1]))
    # img = cv2.imread(path)
    # cv2.imwrite(os.path.join(id_folder, path.split('/')[-1]),img)