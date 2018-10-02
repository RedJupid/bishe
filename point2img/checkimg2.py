import numpy as np
import cv2
import h5py


f = h5py.File('img_data_test1.h5', 'r')
name = []
f2 = open("../data/modelnet40_ply_hdf5_2048/shape_names.txt","r")
line = f2.readline()
while line:
    name.append(line)
    line = f2.readline()
print(f['data'])
print(f['label'])
# print(f['label'][0])

# 显示16*8张图片
img = np.zeros((64,64),dtype=np.uint8)

#第几组数据
group = 3

for j in range(0,8):
    na = []
    nn = []
    for i in range(0,16):
        currentN = group*128+j*16+i
        if currentN >= len(f['label'][0])-1:
            currentN = len(f['label'][0])-1
        #获取模型的名字
        stri = name[int(f['label'][0][currentN])]
        stri = stri[0:-1]
        #生成一张空的图片
        blank = np.zeros((32,64),dtype=np.uint8)
        cv2.putText(blank, stri, (0, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        #得到模型的序号
        nn = str(currentN)
        cv2.putText(blank, nn, (0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        img = f['data'][currentN]
        if i%16 == 0:
            X = f['data'][currentN]
            Xblank = blank
        else:
            X = np.hstack((X,img))
            Xblank = np.hstack((Xblank,blank))
    if j%8 == 0:
        Y = X
    else:
        Y = np.vstack((Y,X))
    Y = np.vstack((Y,Xblank))
# cv2.imwrite('result.jpg',Y)
cv2.namedWindow("Image2")
cv2.imshow("Image2", Y)
cv2.waitKey(0)


f.close()
