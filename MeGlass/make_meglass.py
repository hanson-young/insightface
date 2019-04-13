import os, shutil
import random
import sys
from scipy import misc
gallery_glass_flie = 'test/gallery_black_glass.txt'
gallery_nonglass_flie = 'test/gallery_no_glass.txt'
probe_glass_flie = 'test/probe_black_glass.txt'
probe_nonglass_flie = 'test/probe_no_glass.txt'

meglass_filelists=['test/gallery_black_glass.txt',
                    'test/gallery_no_glass.txt',
                    'test/probe_black_glass.txt',
                    'test/probe_no_glass.txt',
                   ]

input_dir = '/media/handsome/data2/DataSet/FaceVerification/MeGlass_ori'
output_dir = '/media/handsome/data2/DataSet/FaceVerification/MeGlass_300'
for idy, meglass_file in enumerate(meglass_filelists):
    print("parse meglass_file:%s", meglass_file)
    with open(meglass_file,'r') as f:

        for idx, item in enumerate(f.readlines()):
            part = item.strip().split('@')
            target_dir = os.path.join(output_dir, part[0] + '@' + part[1])

            src_path = os.path.join(input_dir,item.strip())
            if not os.path.exists(src_path):
                continue

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            target_file = os.path.join(target_dir, part[2])
            shutil.copyfile(src_path,target_file)
            print("copy %s --> %s"%(src_path, target_file))
            if not os.path.exists(src_path):
                print("copy file Error!")


meglass_folders = os.listdir(output_dir)
random.shuffle(meglass_folders)

retrain_folder = meglass_folders[:300]
delete_folder = meglass_folders[300:]

for i in range(len(delete_folder)):
    shutil.rmtree(os.path.join(output_dir, delete_folder[i]))
    print("remove delete_folder: %s successfully!" % delete_folder[i])