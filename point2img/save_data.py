import provider
import numpy as np
import ctypes as ct
import cv2
import sys
import xd
import h5py

if __name__=='__main__':

    data,label = provider.loadDataFile("../data/modelnet40_ply_hdf5_2048/ply_data_test1.h5")
    name = []
    f = open("../data/modelnet40_ply_hdf5_2048/shape_names.txt","r")
    line = f.readline()
    while line:
        name.append(line)
        line = f.readline()
    f.close()
    #打印所有标签名字
    # print(name)
    for i in range(0,len(label)):
        print(i)
        n_model = i
        # 打印第n_model个模型的标签
        print(name[int(label[n_model])])
        #print(len(data[n_model]))
        # for points in data[n_model]:
        #     print(points)
        # print(data[n_model])
        data_model = np.array(data[n_model])
        # data_model = xd.rotate(data_model, -90, 1, 0, 0)
        # data_model = xd.rotate(data_model, 90, 0, 1, 0)
        # data_model = xd.rotate(data_model, 90, 0, 0, 1)
        labe = str(name[int(label[i])])
        if labe == 'airplane\n' or labe == 'bathtub\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, 180, 0, 1, 0)
        elif labe == 'bed\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, 270, 0, 1, 0)
        elif labe == 'bench\n' or labe == 'bookshelf\n' or labe == 'bottle\n' or labe == 'bowl\n' or labe == 'chair\n':
            data_model = xd.rotate(data_model, 90, 1, 0, 0)
            data_model = xd.rotate(data_model, 90, 0, 1, 0)
            data_model = xd.rotate(data_model, 90, 1, 0, 0)
        elif labe == 'car\n' or labe == 'cone\n' or labe == 'cup\n':
            data_model = xd.rotate(data_model, 90, 1, 0, 0)
            data_model = xd.rotate(data_model, 90, 0, 1, 0)
        elif labe == 'curtain\n' or labe == 'xbox\n':
            data_model = xd.rotate(data_model, 0, 0, 1, 0)
        elif labe == 'door\n':
            data_model = xd.rotate(data_model, 180, 1, 0, 0)
            data_model = xd.rotate(data_model, -90, 0, 0, 1)
        elif labe == 'guitar\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, 180, 0, 1, 0)
        elif labe == 'keyboard\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, 180, 0, 1, 0)
            data_model = xd.rotate(data_model, 90, 0, 0, 1)
        elif labe == 'lamp\n' or labe == 'mantel\n' or labe == 'tv_stand\n' or labe == 'vase\n':
            data_model = xd.rotate(data_model, 90, 0, 0, 1)
        elif labe == 'laptop\n':
            data_model = xd.rotate(data_model, -90, 0, 1, 1)
        elif labe == 'piano\n' or labe == 'range_hood\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, -90, 0, 1, 0)
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
        elif labe == 'radio\n':
            data_model = xd.rotate(data_model, -90, 0, 1, 0)
            data_model = xd.rotate(data_model, 90, 1, 0, 0)
        elif labe == 'sink\n':
            data_model = xd.rotate(data_model, 90, 1, 0, 0)
        elif labe == 'wardrobe\n':
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, 90, 0, 1, 0)
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
        else:
            data_model = xd.rotate(data_model, -90, 1, 0, 0)
            data_model = xd.rotate(data_model, -90, 0, 1, 0)

        b = np.zeros([2048,3])+32#像素为64*64
        v = np.multiply(data_model,b)
        v = np.trunc(v)+32

        img = np.zeros((64,64),dtype=np.uint8)
        # print(np.shape(img))
        #投影到xy平面
        for i in v:
            if i[2]>img[int(i[0]),int(i[1])]:
                img[int(i[0]),int(i[1])] = int(i[2])
        #投影到yz平面
        # for i in v:
        #     if i[0]>img[int(i[2]),int(i[1])]:
        #         img[int(i[2]),int(i[1])] = int(i[0])
        #投影到xz平面
        # for i in v:
        #     if i[1]>img[int(i[0]),int(i[2])]:
        #         img[int(i[0]),int(i[2])] = int(i[1])
        img = cv2.equalizeHist(img)
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        #保存模型
        xd.save_h5('img_data_test1.h5',img,n_model,label)


