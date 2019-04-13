#encoding=utf-8
import sys
import os

base_inputdir = '/media/handsome/data2/DataSet/FaceVerification/HSVD/clearn-face0816-112x112/base'
recg_inputdir = '/media/handsome/data2/DataSet/FaceVerification/HSVD/clearn-face0816-112x112/recg'

pairs_file = '/media/handsome/data2/DataSet/FaceVerification/HSVD/clearn-face0816-112x112/bin/hsvd_pairs.txt'


base_list = []
for idx, itemx in enumerate(os.listdir(base_inputdir)):
    base_folder = os.path.join(base_inputdir, itemx)
    each_folder_imgs = []
    for idy, itemy in enumerate(os.listdir(base_folder)):
        each_folder_imgs.append("base/" + itemx + "/" + itemy)
    base_list.append(each_folder_imgs)

print(base_list)
recg_list = []
for idx, itemx in enumerate(os.listdir(recg_inputdir)):
    recg_folder = os.path.join(recg_inputdir, itemx)
    each_folder_imgs = []
    for idy, itemy in enumerate(os.listdir(recg_folder)):
        each_folder_imgs.append("recg/" + itemx + "/" + itemy)
    recg_list.append(each_folder_imgs)
print(recg_list)


hsvd_lines = []
for i in range(len(base_list)):
    for j in range(len(base_list[i])):
        for m in range(len(recg_list)):
            for n in range(len(recg_list[m])):
                label = '0'
                if base_list[i][j].split('/')[1] == recg_list[m][n].split('/')[1]:
                    label = '1'
                a = base_list[i][j]
                b = recg_list[m][n]

                line = "%s\t%s\t%s\n" % (a, b, label)
                hsvd_lines.append(line)

# print("before choose:%s"%len(hsvd_lines))
# hsvd_lines = list(set(hsvd_lines))

print("after choose:%s"%len(hsvd_lines))

with open(pairs_file,'w') as f:
    # for i in range(len(meglass_lines)):
    f.writelines(hsvd_lines)