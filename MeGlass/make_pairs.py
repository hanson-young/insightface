import sys
import os

input_dir = '/media/handsome/data2/DataSet/FaceVerification/MeGlass_112x112'
pairs_file = '/media/handsome/data2/DataSet/FaceVerification/bin/meglass_pairs.txt'

meglass_list = []

for idx, itemx in enumerate(os.listdir(input_dir)):
    meglass_folder = os.path.join(input_dir, itemx)
    each_folder_imgs = []
    for idy, itemy in enumerate(os.listdir(meglass_folder)):
        each_folder_imgs.append(itemx + "/" + itemy)
    meglass_list.append(each_folder_imgs)

# print(meglass_list)
meglass_lines = []

for i in range(len(meglass_list)):
    for j in range(len(meglass_list[i])):
        for m in range(len(meglass_list)):
            for n in range(len(meglass_list[m])):
                if meglass_list[i][j] == meglass_list[m][n]:
                    continue
                a = meglass_list[i][j]
                b = meglass_list[m][n]
                if cmp(a, b) == 0:
                    continue
                elif cmp(a,b) == 1:
                    tmp = a
                    a = b
                    b = tmp
                else:
                    pass

                line = "%s\t%s\t%s\n"%(a, b, '1' if i == m else '0')
                meglass_lines.append(line)

print("before choose:%s"%len(meglass_lines))
meglass_lines = list(set(meglass_lines))

print("after choose:%s"%len(meglass_lines))

with open(pairs_file,'w') as f:
    # for i in range(len(meglass_lines)):
    f.writelines(meglass_lines)