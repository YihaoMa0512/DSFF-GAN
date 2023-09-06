from os.path import join
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

HE_PATH='train/big/A'
Ki67_PATH='train/big/B'
for he_name in os.listdir(HE_PATH):
    for ki67_name in os.listdir(Ki67_PATH):
        if he_name[0:2] in ki67_name:
            he_image = Image.open(HE_PATH + '/' + he_name)
            ki67_image = Image.open(Ki67_PATH + '/' + ki67_name)
            w = int(he_image.size[0]//256)
            h = int(he_image.size[1]//256)
            f=0
            for i in tqdm(range(w)):
                for j in range(h):
                    cut_img = he_image.crop((i * 256, j * 256, (i + 1) * 256, (j + 1) * 256))
                    cut_img_array=np.array(cut_img)
                    if (np.sum(cut_img_array >= 240)) / (256 * 256 * 3)<=0.97 :
                        cut_img1 = ki67_image.crop((i * 256, j * 256, (i + 1) * 256, (j + 1) * 256))
                        cut_img_array1 = np.array(cut_img1)
                        if (np.sum(cut_img_array1 == 0)) / (256 * 256 * 3) <= 0.05:
                            # cut_img.save(join('datasets/he2ki67/testA', "HE" +str(f).zfill(2)+ str(i).zfill(3) + '_' + str(j).zfill(3) + ".png"))
                            # cut_img1.save(join('datasets/he2ki67/testB',"Ki67" +str(f).zfill(2)+ str(i).zfill(3) + '_' + str(j).zfill(3) + ".png"))
                            cut_img.save(join('train/A',
                                              str(he_name[0:2]).zfill(2) + str(i).zfill(3) + '_' + str(j).zfill(3) + ".png"))
                            cut_img1.save(join('train/B',
                                               str(he_name[0:2]).zfill(2) + str(i).zfill(3) + '_' + str(j).zfill(3) + ".png"))
                            f=f+1
            print(he_name[0:2],f)