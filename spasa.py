# -*- coding: utf-8 -*-
"""SpaSA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xVUFgSLOSfgsVIU-0RmWXhedNrFzb9dr
"""


import argparse
import scipy.io as sio
from sklearn import metrics, preprocessing
from averageAccuracy import AA_andEachClassAccuracy
from zeroPadding import zeroPadding_3D
import numpy as np
import math

from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
import random

from sklearn.decomposition import PCA
# from skimage.color import label2rgb
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_PU', help='input data dir', default='F:\\NAS-LAFFN\\UP\\dataset\\PaviaU.mat')
parser.add_argument('--input_gt_PU', help='input gt dir', default='F:\\NAS-LAFFN\\UP\\dataset\\PaviaU_gt.mat')
parser.add_argument("--n_classes", type=int, default=9, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=103, help="number of image channels")

opt = parser.parse_args(args=[])

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len+1)]
    return selected_patch

def sampling1(Train_size,Validation_pro, groundTruth):  # divide dataset into train and test datasets
    np.random.seed(0)
    labels_loc = {}
    train = {}
    test = {}
    validation = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        Validation_size = math.ceil(Validation_pro * len(indices))
        labels_loc[i] = indices
        train[i] = labels_loc[i][:Train_size]
        validation[i] = labels_loc[i][Train_size:Train_size + Validation_size]
        test[i] = labels_loc[i][Train_size:]
    train_indices = []
    test_indices = []
    val_indices = []
    for i in range(m):
        train_indices += train[i]
        val_indices += validation[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    return train_indices,val_indices, test_indices


def sampling2(proptionTrain, proptionVal, groundTruth):
    np.random.seed(0)
    labels_loc = {}
    train = {}
    validation = {}
    test = {}
    m = max(groundTruth)
    print(m)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_train = math.ceil(proptionTrain * len(indices))
        nb_Val = math.ceil(proptionVal * len(indices))
        train[i] = indices[:nb_train]
        test[i] = indices[nb_train:]
        np.random.shuffle(indices)
        validation[i] = indices[nb_train:nb_train + nb_Val]
    train_indices = []
    test_indices = []
    val_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        val_indices += validation[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    return train_indices,val_indices, test_indices

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
      #正态分布，mean=0, std=0.02
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
      #正态分布，mean=1.0, std=0.02
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        #初始化整个矩阵为常数0
        torch.nn.init.constant_(m.bias.data, 0.0)

def generate_real_samples(n_samples):
  images_spa, labels =  X_train_spa, y_train
  ix = randint(0, images_spa.shape[0], n_samples)
  X_spa, labels = images_spa[ix], labels[ix]
  X_spa = Variable(X_spa.type(FloatTensor))
  labels = Variable(torch.from_numpy(labels).type(LongTensor))
  return [X_spa, labels]

def summarize_performance(step, d_model):
  print(step)
  filename_d = '3D-models/Dmodel_%04d.pth' % (step+1)
#  filename_g = '3D-models/Gmodel_%04d.pth' % (step+1)
  torch.save(d_model.state_dict(), filename_d)
#  torch.save(g_model.state_dict(), filename_g)

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.列向量标签值转化为二维矩阵

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)  # isinstance：判断类型 若是同一类返回true assert：检查条件，不符合就终止程序
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1  # np.max:取最大值 这里计算已知标签类别数加上未分类标签
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))  # np.zeros:返回一个给定形状和类型的用0填充的数组
    result[np.arange(len(vector)), vector] = 1  # 对全零矩阵result进行OneHot编码 np.arange：一个参数时，参数值为终点，起点取默认值0，步长取默认值1
    return result.astype(int)  # astype：强制类型转换

#load dataset
import scipy.io as sio
mat_data = sio.loadmat(opt.input_data_PU)
mat_gt = sio.loadmat(opt.input_gt_PU)
data_PU = mat_data['paviaU']
gt_PU = mat_gt['paviaU_gt']
print(data_PU.shape)
print(gt_PU.shape)
Height, Width, Band = data_PU.shape[0], data_PU.shape[1], data_PU.shape[2]
PATCH_LENGTH = 3

def overall_sample_divide(data_PU, gt_PU, Band, PATCH_LENGTH):
    TOTAL_SIZE = 42776  # 总标记样本
    TRAIN_SIZE = 432  # 总训练样本
    VAL_SIZE = 1286  # 总验证样本
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    INPUT_DIMENSION_CONV = opt.channels
    Height, Width, Band = data_PU.shape[0], data_PU.shape[1], data_PU.shape[2]
    data = data_PU.reshape(-1,data_PU.shape[2])
    gt = gt_PU.reshape(-1)

    #标准化
    data = preprocessing.scale(data)

    #PCA
    pca = PCA(n_components = INPUT_DIMENSION_CONV)
    principalComponents = pca.fit_transform(data)

    data_new = data.reshape(data_PU.shape[0], data_PU.shape[1], opt.channels)
    data_ = principalComponents.reshape(data_PU.shape[0], data_PU.shape[1], INPUT_DIMENSION_CONV)
    whole_data = data_

    # zeroPADDING
    padded_data_spe = zeroPadding_3D(data_new, 0)
    padded_data_spa = zeroPadding_3D(whole_data, PATCH_LENGTH)

    print('whole_data : ', whole_data.shape)
    print('padded_data : ', padded_data_spa.shape)

    train_indices, val_indices, test_indices = sampling2(0.01, 0.03, gt) # 按比例取训练样本,1%训练样本,3%验证样本

    y_train = gt[train_indices] - 1
    y_val = gt[val_indices] - 1
    y_test = gt[test_indices] - 1

    print('y_train : ',len(train_indices))
    print('y_val : ',len(val_indices))
    print('y_test : ',len(test_indices))

    train_spa = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH+1, 2 * PATCH_LENGTH+1, Band))
    val_spa = np.zeros((VAL_SIZE, 2 * PATCH_LENGTH+1, 2 * PATCH_LENGTH+1, Band))
    test_spa = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH+1, 2 * PATCH_LENGTH+1, Band))

    #spa
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
      train_spa[i] = selectNeighboringPatch(padded_data_spa, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    val_assign = indexToAssignment(val_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(val_assign)):
      val_spa[i] = selectNeighboringPatch(padded_data_spa, val_assign[i][0], val_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
      test_spa[i] = selectNeighboringPatch(padded_data_spa, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    y_train = y_train
    y_val = y_val
    y_test = y_test

    y_train = convertToOneHot(y_train, num_classes=9)
    y_val = convertToOneHot(y_val, num_classes=9)
    y_test = convertToOneHot(y_test, num_classes=9)

    return TRAIN_SIZE, train_spa, val_spa, test_spa, \
           y_train , y_val, y_test, padded_data_spa, TOTAL_SIZE, train_indices, val_indices, test_indices
