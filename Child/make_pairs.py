#encoding=utf-8
import sys
import os

base_inputdir = '/media/yj/hanson/face-recognition/china_chip/align112x112'

pairs_file = '/media/yj/hanson/face-recognition/china_chip/bin/chip_pairs.txt'


base_list = []
for idx, itemx in enumerate(os.listdir(base_inputdir)):
    base_folder = os.path.join(base_inputdir, itemx)
    each_folder_imgs = []
    for idy, itemy in enumerate(os.listdir(base_folder)):
        each_folder_imgs.append("align112x112/" + itemx + "/" + itemy)
    base_list.append(each_folder_imgs)

print(base_list)

hsvd_lines = []
for i in range(len(base_list)):
    for j in range(len(base_list)):
        if i == j:
            continue

        a = base_list[i][0]
        b = base_list[j][0]

        line = "%s\t%s\t%s\n" % (a, b, 0)
        hsvd_lines.append(line)

# print("before choose:%s"%len(hsvd_lines))
# hsvd_lines = list(set(hsvd_lines))

print("after choose:%s"%len(hsvd_lines))

with open(pairs_file,'w') as f:
    # for i in range(len(meglass_lines)):
    f.writelines(hsvd_lines)