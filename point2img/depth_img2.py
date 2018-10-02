import provider
import numpy as np
import ctypes as ct
import cv2
import sys
import xd
import h5py
from depth_img import *
import re

if __name__=='__main__':

    data,label = provider.loadDataFile("../data/modelnet40_ply_hdf5_2048/ply_data_test1.h5")
    name = []
    f = open("../data/modelnet40_ply_hdf5_2048/shape_names.txt","r")
    line = f.readline()
    while line:
        name.append(line)
        line = f.readline()
    f.close()
    while 1:
        n_model = input('请输入模型编号：')
        n_model = int(n_model)
        # 打印第n_model个模型的标签
        print('该模型为：'+name[int(label[n_model])][0:-1])

        data_model = np.array(data[n_model])
        # data_model = xd.rotate(data_model, 90, 1, 0, 0)
        # data_model = xd.rotate(data_model, 90, 0, 1, 0)
        # data_model = xd.rotate(data_model, 90, 0, 0, 1)
        #点云可视化
        cv2.namedWindow('show3d')
        cv2.moveWindow('show3d',0,0)
        cv2.setMouseCallback('show3d',onmouse)
        showpoints(data_model)

        stri = input('请输入旋转角度：')
        arr = re.split(' ', stri)
        print(arr)
        for i in arr:
            if i == '':
                break
            if i[0] == 'x':
                data_model = xd.rotate(data_model, int(i[1:]), 1, 0, 0)
            elif i[0] == 'y':
                data_model = xd.rotate(data_model, int(i[1:]), 0, 1, 0)
            elif i[0] == 'z':
                data_model = xd.rotate(data_model, int(i[1:]), 0, 0, 1)

        b = np.zeros([2048, 3]) + 32  # 像素为64*64
        v = np.multiply(data_model, b)
        v = np.trunc(v) + 32
        img = np.zeros((64, 64), dtype=np.uint8)
        # 投影到xy平面
        for i in v:
            if i[2] > img[int(i[0]), int(i[1])]:
                img[int(i[0]), int(i[1])] = int(i[2])
        img = cv2.equalizeHist(img)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        #改变模型
        xd.change_h5('img_data_test1.h5',img,n_model)
        print('模型已更新')
        cv2.destroyAllWindows()

