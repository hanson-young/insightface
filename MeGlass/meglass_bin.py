import pickle
import os

input_dir = '/media/handsome/data2/DataSet/FaceVerification/MeGlass_112x112'
output_bin = '/media/handsome/data2/DataSet/FaceVerification/bin/MeGlass.bin'

meglass_list = []
meglass_lines = []
for idx, itemx in enumerate(os.listdir(input_dir)):
    meglass_folder = os.path.join(input_dir, itemx)
    each_folder_imgs = []
    for idy, itemy in enumerate(os.listdir(meglass_folder)):
        each_folder_imgs.append(itemx + "/" + itemy)
    meglass_list.append(each_folder_imgs)

print(len(meglass_list))
for i in range(len(meglass_list)):
    for j in range(len(meglass_list[i])):
        meglass_lines.append(meglass_list[i][j])


print(len(meglass_lines))
print(meglass_lines)

meglass_bins = []
i = 0
for path in meglass_lines:
    path = os.path.join(input_dir,path)
    with open(path, 'rb') as fin:
        _bin = fin.read()
        meglass_bins.append(_bin)
        i+=1
        if i%100==0:
            print('loading meglass', i)

with open(output_bin, 'wb') as f:
  pickle.dump((meglass_bins, meglass_lines), f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(output_bin, 'r') as f:
#     bin = pickle.load(f)
#     print(bin)