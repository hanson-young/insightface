import sys
import os
import shutil

input_dir = '/media/handsome/data2/DataSet/FaceVerification/HSVD/face0816/base'
output_dir = '/media/handsome/data2/DataSet/FaceVerification/HSVD/clearn-face0816/base'

hsvd_list = []

for idx, itemx in enumerate(os.listdir(input_dir)):
    hsvd_folder = os.path.join(input_dir, itemx)
    each_folder_imgs = []
    clearn_hsvd_folder = os.path.join(output_dir, itemx)
    if not os.path.exists(clearn_hsvd_folder):
        os.makedirs(clearn_hsvd_folder)
    for idy, itemy in enumerate(os.listdir(hsvd_folder)):
        base_image = os.path.join(hsvd_folder, itemy)
        clearn_base_image = os.path.join(clearn_hsvd_folder, itemy)
        # shutil.move(base_image,clearn_base_image)
        shutil.copyfile(base_image,clearn_base_image)
        rename = clearn_base_image
        if '.png' in clearn_base_image:
            rename = clearn_base_image.replace('.png','.jpg')
            os.rename(clearn_base_image, rename)

        each_folder_imgs.append(itemx + "/" + rename)
        hsvd_list.append(each_folder_imgs)

print("list:%d"%len(hsvd_list))
# with open(base_file,'w') as f:
#     # for i in range(len(meglass_lines)):
#     f.writelines(hsvd_list)