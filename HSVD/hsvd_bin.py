import pickle
import os

input_dir = '/media/yj/hanson/face-recognition/HSVD/clearn-face0816-112x112'
base_inputdir = '/media/yj/hanson/face-recognition/HSVD/clearn-face0816-112x112/base'
recg_inputdir = '/media/yj/hanson/face-recognition/HSVD/clearn-face0816-112x112/recg'

hsvd_bin = '/media/yj/hanson/face-recognition/HSVD/clearn-face0816-112x112/bin/hsvd.bin'


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


hsvd_list = base_list + recg_list

hsvd_names = []

print(len(hsvd_list))
for i in range(len(hsvd_list)):
    for j in range(len(hsvd_list[i])):
        hsvd_names.append(hsvd_list[i][j])


print(len(hsvd_names))


hsvd_bins = []
i = 0
for path in hsvd_names:
    path = os.path.join(input_dir,path)
    with open(path, 'rb') as fin:
        _bin = fin.read()
        hsvd_bins.append(_bin)
        i+=1
        if i%100==0:
            print('loading hsvd', i)

with open(hsvd_bin, 'wb') as f:
  pickle.dump((hsvd_bins, hsvd_names), f, protocol=pickle.HIGHEST_PROTOCOL)

with open(hsvd_bin, 'r') as f:
    bin = pickle.load(f)
    # print(bin)