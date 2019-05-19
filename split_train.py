# AHLT
# Olga Valls
# 1. Split the training set into train (90%) and validation (10%)

import os
import shutil

path = 'data/Train'
if not os.path.exists('data/Val'):
    os.mkdir('data/Val')

folder_list = os.listdir(path)
print('Folder list: {}'.format(folder_list))

for f in folder_list:
    if (f[0] != '.'):
        print('- Folder: {}'.format(f))
        full_path = os.path.join(path, f)

        files = os.listdir(full_path)
        num_files = len(files)
        # 10% of the training data, for each folder, goes to validation set
        num_val = int(0.1 * num_files)
        print('Num files: {} --> num validation: {}'.format(num_files, num_val))

        for i, file in enumerate(files):
            #print('..file: {}'.format(file))
            if (i > 0 and i % 10 == 0):
                shutil.move(os.path.join(full_path,file),'data/Val')

