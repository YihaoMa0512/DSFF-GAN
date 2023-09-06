from os.path import join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def img_sift(small_image1,small_image2,big_image1,big_image2,sigma,ratio,f):
    start = time.time()
    bigimg=Image.fromarray((small_image1*255).astype('uint8')).convert('RGB')
    bigimg.save(join('CUT-img/SIFT_OUT', str(f).zfill(2) +"_Imagein.jpg"))
    sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=None,
                           contrastThreshold=None, edgeThreshold=None, sigma=sigma)
    #  使用SIFT查找关键点key points和描述符descriptors
    image1 = cv2.normalize(small_image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    image2 = cv2.normalize(small_image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    reimage1 = cv2.normalize(big_image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    reimage2 = cv2.normalize(big_image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    end = time.time()
    print("特征点提取&生成描述运行时间:%.2f秒" % (end - start))

    kp_image1 = cv2.drawKeypoints(image1, kp1, None)
    kp_image2 = cv2.drawKeypoints(image2, kp2, None)

    plt.figure()
    plt.imshow(kp_image1)
    # plt.savefig( join('data/sift_out', str(f) +"kp_image1.jpg"))

    plt.figure()
    plt.imshow(kp_image2)
    # plt.savefig(join('data/sift_out', str(f) +"kp_image2.jpg"))
    print("关键点数目:", len(kp1))

    for i in range(2):
        print("关键点", i)
        print("数据类型:", type(kp1[i]))
        print("关键点坐标:", kp1[i].pt)
        print("邻域直径:", kp1[i].size)
        print("方向:", kp1[i].angle)
        print("所在的图像金字塔的组:", kp1[i].octave)
        print("================")

    #  查看描述
    print("描述的shape:", des1.shape)
    for i in range(2):
        print("描述", i)
        print(des1[i])


    #  计算匹配点匹配时间
    start = time.time()

    #  K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])
    end = time.time()
    print("匹配点匹配运行时间:%.2f秒" % (end - start))

    matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=2)
    matches = Image.fromarray(matches)
    matches.save(join('CUT-img/SIFT_OUT',str(f).zfill(2) + 'matches.jpg'))

    # plt.savefig(join('results/HE2SMA_pretrained/sift',str(i).zfill(3) + '_' + str(j).zfill(3) + 'matches.jpg'))
    print("匹配对的数目:", len(good_matches))

    #  单应性矩阵有八个参数，每一个对应的像素点可以产生2个方程(x一个，y一个)，那么需要四个像素点就能解出单应性矩阵
    if len(good_matches) > 4:
        for i in range(2):
            print("匹配", i)
            print("数据类型:", type(good_matches[i][0]))
            print("描述符之间的距离:", good_matches[i][0].distance)
            print("查询图像中描述符的索引:", good_matches[i][0].queryIdx)
            print("目标图像中描述符的索引:", good_matches[i][0].trainIdx)
            print("================")
        #  计算匹配时间
        start = time.time()
        ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        #  单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        S = np.array([[1/16,0,0],[0,1/16,0],[0,0,1]])
        s = np.linalg.inv(S)
        H1 = np.dot(np.dot(s, H), S)



        imgOut_array = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        imgOut1_array = cv2.warpPerspective(reimage2, H1, (reimage1.shape[1], reimage1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        end = time.time()
        print("匹配运行时间:%.2f秒" % (end - start))
        imgOut = Image.fromarray(imgOut_array)
        imgOut.save(join('CUT-img/SIFT_OUT', str(f).zfill(2) + '_imgOut.jpg'))
        # imgOut1 = Image.fromarray(imgOut1_array)
        # imgOut1.save(join('CUT-img/SIFT',str(f).zfill(2) + '_imgOut1.png'))




if __name__ == '__main__':
    image2 = mpimg.imread("CUT-img/small_Ki67/01 Ki67small.png")
    image1 = mpimg.imread("CUT-img/small_HE/01 HEsmall.png")
    reimage2 = mpimg.imread("CUT-img/Ki67/01 Ki67.png")
    reimage1 = mpimg.imread("CUT-img/HE/01 HE.png")
    img_sift(image1, image2, reimage1, reimage2,2.5, 0.8, 1)

    #

