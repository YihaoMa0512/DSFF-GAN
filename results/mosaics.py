# By mama
# TIME 2022/10/25   13:05
from tqdm import tqdm
import os
from PIL import Image
from os.path import join

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
def mosaics(mode):
    IMAGE_SIZE = 512 # 每张小图片的大小
    big_image_path='../datasets1/Ki67-002/Ki67/TrainValAB/valBre'
    small_image_path = 'data/big/'+mode
    big_image_names=os.listdir(big_image_path)
    # 获取图片集地址下的所有图片名称

    small_image_names =os.listdir(small_image_path)
    for big_image_name in big_image_names:
        big_image = Image.open(big_image_path + '/' + big_image_name)
        # img_lie = big_image.size[0] // IMAGE_SIZE  # 一共有几列
        # img_hang = big_image.size[1] // IMAGE_SIZE  # 一共有几行
        # 简单的对于参数的设定和实际图片集的大小进行数量判断
        # if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
        #     raise ValueError("合成图片的参数和要求的数量不能匹配！")
        to_image = Image.new('RGB', (big_image.size[0] , big_image.size[1] ), 'white')
        # to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
        for small_image_name in tqdm(small_image_names):

            if int(big_image_name[0:4])==int(small_image_name[0:4]):
                # 循环遍历，把每张图片按顺序粘贴到对应位置上
                small_image = Image.open(small_image_path + '/' + small_image_name)
                m=int(small_image_name[5:6])
                n=int(small_image_name[7:8])
                to_image.paste(small_image, (m* IMAGE_SIZE, n * IMAGE_SIZE))
        if not os.path.exists('data/merge/'+mode):
            os.makedirs('data/merge/'+mode)
        to_image.save(join('data/merge/'+mode,big_image_name))

        print(big_image_name)
# pinjie('PC')
mosaics('dsff1')
# pinjie('cyclegan_Resnet')
