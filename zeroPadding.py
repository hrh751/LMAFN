import numpy as np

def zeroPadding_2D(old_matrix, pad_length):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length),(pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

#def zeroPadding_1D(old_vector, pad_length):


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values = (10, 10))
    return new_matrix

def zeroPadding_1D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((0, pad_length)), 'constant', constant_values=0)
    return new_matrix

def image_pad(data, r):
    if len(data.shape) == 3:  # len：返回对象长度或者项目个数  shape:获取矩阵的形状;
        data_new = np.lib.pad(data, ((r, r), (r, r), (0, 0)),
                              'symmetric')  # symmetric:对称填充  np.lib.pad:填充函数，由低维到高维填充 上下左右前后
        return data_new
    if len(data.shape) == 2:
        data_new = np.lib.pad(data, r, 'constant', constant_values=0)  # constant：连续填充相同的值，每个轴可以分别指定填充值
        return data_new

