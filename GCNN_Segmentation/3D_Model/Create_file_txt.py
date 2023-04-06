import os
from natsort import natsorted

data_path = 'C:/Users/sonbu/Desktop/LVTN/Dataset_3D/LIDC/LIDC-IDRI-0195/nodule-1/images'
msk_path = 'C:/Users/sonbu/Desktop/LVTN/Dataset_3D/LIDC/LIDC-IDRI-0195/nodule-1/mask-0'

files_path_data = [os.path.join(data_path,file) for file in (os.listdir(data_path))]
files_path_data = natsorted(files_path_data)

files_path_msk = [os.path.join(msk_path,file) for file in (os.listdir(msk_path))]
files_path_msk = natsorted(files_path_msk)

f1 = open("C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_LIDC.txt", "w")
f2 = open("C:/Users/sonbu/Desktop/LVTN/3D_Model/demo_label_LIDC.txt", "w")

for file_path_data in files_path_data:
    print(file_path_data)
    f1.write("{}\n".format(file_path_data))
f1.close()

for file_path_msk in files_path_msk:
    print(file_path_msk)
    f2.write("{}\n".format(file_path_msk))
f2.close()