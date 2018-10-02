import numpy as np
import h5py
import math

def rotate(model,angle,x,y,z):
    model = np.array(model)
    sina = math.sin(angle/180*math.pi)
    cosa = math.cos(angle/180*math.pi)
    row = model.shape[0]
    xx = model[:,0]
    yy = model[:,1]
    zz = model[:,2]
    xx = np.reshape(xx,(row,1))
    yy = np.reshape(yy, (row, 1))
    zz = np.reshape(zz, (row, 1))
    if z >0:
        xxx = np.multiply(xx,cosa) - np.multiply(yy,sina)
        yyy = np.multiply(xx,sina) + np.multiply(yy,cosa)
        zzz = zz
    elif y>0:
        xxx = np.multiply(xx, cosa) - np.multiply(zz, sina)
        yyy = yy
        zzz = np.multiply(xx, sina) + np.multiply(zz, cosa)
    elif x>0:
        xxx = xx
        yyy = np.multiply(yy, cosa) - np.multiply(zz, sina)
        zzz = np.multiply(yy, sina) + np.multiply(zz, cosa)
    x = np.hstack((xxx,yyy))
    x = np.hstack((x,zzz))
    return x

def save_h5(filename,model,times,label):
    if times == 0:
        h5f = h5py.File(filename, 'w')
        dataset = h5f.create_dataset("data", (1, 64, 64),
                                     maxshape=(None, 64, 64),
                                     # chunks=(1, 1000, 1000),
                                     dtype=np.uint8)
        dataset2 = h5f.create_dataset("label",(1,len(label),1))
        dataset2[0] = label
    else:
        h5f = h5py.File(filename, 'a')
        dataset = h5f['data']
    # 关键：这里的h5f与dataset并不包含真正的数据，
    # 只是包含了数据的相关信息，不会占据内存空间
    #
    # 仅当使用数组索引操作（eg. dataset[0:10]）
    # 或类方法.value（eg. dataset.value() or dataset.[()]）时数据被读入内存中
    a = np.zeros((1, 64, 64)).astype(np.uint8)
    a[0] = model
    # 调整数据预留存储空间（可以一次性调大些）
    dataset.resize([times*1+1, 64, 64])
    # 数据被读入内存
    dataset[times*1:times*1+1] = a
    # print(sys.getsizeof(h5f))
    h5f.close()

def change_h5(filename,model,n):
    h5f = h5py.File(filename, 'a')
    dataset = h5f['data']
    dataset[n*1:n*1+1] = model