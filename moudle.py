from __future__ import print_function
import tensorflow as tf
import math
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from spasa import Band, PATCH_LENGTH
import numpy as np

patch_size = PATCH_LENGTH * 2 + 1
spe_band = 64
spa_band = 32
n_classes = 9
num_band = Band
is_training = tf.placeholder(dtype=tf.float32)  # 定义BN是否是训练状态

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

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def BN_layer(value, is_training=False, name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果

    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training == 1:
        # 训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, epsilon=0.01, updates_collections=None,
                                            is_training=True)
    else:
        # 测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, epsilon=0.01, updates_collections=None,
                                            is_training=False)


def avg_pool(x):
    '''
    全局平均池化层，使用一个与原有输入同样尺寸的filter进行池化，'SAME'填充方式  池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）

    args；
        x:输入图像 形状为[batch,in_height,in_width,in_channels]
    '''
    return tf.nn.avg_pool2d(x, ksize=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1],
                            padding='VALID')

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W_in, W_out, k):  # 定义卷积层
    W = weight_variable([k, k, W_in, W_out])
    b = bias_variable([W_out])
    layer = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    return layer


def depthwise_conv2d(x, W_in, rate, k):  # 定义深度卷积层
    W = weight_variable([k, k, W_in, rate])
    b = bias_variable([W_in * rate])
    layer = tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], rate=[1, 1], padding='SAME') + b
    return layer

def Inverted_residuals1(x, a, b, k_1, k_2, k_3):
    p_1d = int(x.shape[3])
    p_1 = tf.nn.relu6(BN_layer(conv2d(x, p_1d, a, 1), is_training=is_training))
    depth_1 = depthwise_conv2d(p_1, a, 1, k_1)
    depth_2 = depthwise_conv2d(p_1, a, 1, k_1)
    depth_sum = tf.concat([depth_1, depth_2, p_1], -1)
    depth_sum = tf.nn.relu6(BN_layer(depth_sum, is_training=is_training))
    depth_sum = ECA(depth_sum)
    p_2 = BN_layer(conv2d(depth_sum, 3 * a, b, 1), is_training=is_training)
    if p_1d == b:
        output = tf.add(p_2, x)
    else:
        p_3 = BN_layer(conv2d(x, p_1d, b, 1), is_training=is_training)
        output = tf.add(p_3, p_2)
    return output


def ECA(x, gamma=2, b=1):
    shape = x.get_shape().as_list()
    C = shape[3]
    t = int(abs(math.log(C, 2) + b) / gamma)
    if t % 2 == 1:
        k = t
    else:
        k = t + 1
    pooling = tf.nn.avg_pool(x, [1, shape[1], shape[2], 1], [1, shape[1], shape[2], 1], padding="SAME")
    y = tf.squeeze(pooling, [1])
    y = tf.transpose(y, (0, 2, 1))
    b = bias_variable([C, 1])
    W = weight_variable([k, 1, 1])
    y = tf.nn.conv1d(y, W, 1, 'SAME') + b
    y = tf.transpose(y, (0, 2, 1))
    y = tf.expand_dims(y, 1)
    y = tf.nn.sigmoid(y)
    h_output = x * y

    return h_output

def model_1(net):
    layer_00 = tf.nn.relu6(BN_layer(conv2d(net, num_band, spe_band, 1), is_training=is_training))

    layer_01 = tf.nn.relu6(BN_layer(conv2d(layer_00, spe_band, spe_band, 1), is_training=is_training))
    layer_01 = tf.add(layer_00, layer_01)
    layer_02 = tf.nn.relu6(BN_layer(conv2d(layer_01, spe_band, spe_band, 1), is_training=is_training))
    layer_02 = tf.add(layer_01, layer_02)
    layer_03 = tf.nn.relu6(BN_layer(conv2d(layer_02, spe_band, spe_band, 1), is_training=is_training))
    layer_03 = tf.add(layer_02, layer_03)
    layer_04 = tf.nn.relu6(BN_layer(conv2d(layer_03, spe_band, spe_band, 1), is_training=is_training))
    layer_04 = tf.add(layer_03, layer_04)
    layer_05 = tf.nn.relu6(BN_layer(conv2d(layer_04, spe_band, spe_band, 1), is_training=is_training))
    layer_05 = tf.add(layer_04, layer_05)
    layer_06 = tf.nn.relu6(BN_layer(conv2d(layer_05, spe_band, spe_band, 1), is_training=is_training))
    layer_06 = tf.add(layer_05, layer_06)
    layer_07 = tf.nn.relu6(BN_layer(conv2d(layer_06, spe_band, spe_band, 1), is_training=is_training))
    layer_07 = tf.add(layer_06, layer_07)
    layer_08 = tf.nn.relu6(BN_layer(conv2d(layer_07, spe_band, spe_band, 1), is_training=is_training))
    layer_08 = tf.add(layer_07, layer_08)
    layer_09 = tf.nn.relu6(BN_layer(conv2d(layer_08, spe_band, spe_band, 1), is_training=is_training))
    layer_09 = tf.add(layer_08, layer_09)
    layer_10 = tf.nn.relu6(BN_layer(conv2d(layer_09, spe_band, spe_band, 1), is_training=is_training))
    layer_10 = tf.add(layer_09, layer_10)
    layer_11 = tf.nn.relu6(BN_layer(conv2d(layer_10, spe_band, spe_band, 1), is_training=is_training))
    layer_11 = tf.add(layer_10, layer_11)
    layer_12 = tf.nn.relu6(BN_layer(conv2d(layer_11, spe_band, spe_band, 1), is_training=is_training))
    layer_12 = tf.add(layer_11, layer_12)
    layer_13 = tf.nn.relu6(BN_layer(conv2d(layer_12, spe_band, spe_band, 1), is_training=is_training))
    layer_13 = tf.add(layer_12, layer_13)
    layer_14 = tf.nn.relu6(BN_layer(conv2d(layer_13, spe_band, spe_band, 1), is_training=is_training))
    layer_14 = tf.add(layer_13, layer_14)
    layer_15 = tf.nn.relu6(BN_layer(conv2d(layer_14, spe_band, spe_band, 1), is_training=is_training))
    layer_15 = tf.add(layer_14, layer_15)
    layer_16 = tf.nn.relu6(BN_layer(conv2d(layer_15, spe_band, spe_band, 1), is_training=is_training))
    layer_16 = tf.add(layer_15, layer_16)
    layer_17 = tf.nn.relu6(BN_layer(conv2d(layer_16, spe_band, spe_band, 1), is_training=is_training))
    layer_17 = tf.add(layer_16, layer_17)
    layer_18 = tf.nn.relu6(BN_layer(conv2d(layer_17, spe_band, spe_band, 1), is_training=is_training))
    layer_18 = tf.add(layer_17, layer_18)
    layer_19 = tf.nn.relu6(BN_layer(conv2d(layer_18, spe_band, spe_band, 1), is_training=is_training))
    layer_19 = tf.add(layer_18, layer_19)
    layer_20 = tf.nn.relu6(BN_layer(conv2d(layer_19, spe_band, spe_band, 1), is_training=is_training))
    layer_20 = tf.add(layer_19, layer_20)
    layer_21 = tf.nn.relu6(BN_layer(conv2d(layer_20, spe_band, spa_band, 1), is_training=is_training))
    stage_1 = layer_20

    layer_27 = Inverted_residuals1(layer_21, spa_band, spa_band, 3, 5, 7)
    stage_2 = layer_27

    layer_30 = Inverted_residuals1(stage_2, spa_band, spa_band, 3, 5, 7)
    layer_31 = Inverted_residuals1(layer_30, spa_band, spa_band, 3, 5, 7)
    layer_32 = Inverted_residuals1(layer_31, spa_band, spa_band, 3, 5, 7)
    stage_3 = tf.add(layer_30, layer_32)

    layer_33 = Inverted_residuals1(stage_3, spa_band, spa_band, 3, 5, 7)
    layer_34 = Inverted_residuals1(layer_33, spa_band, spa_band, 3, 5, 7)
    layer_35 = Inverted_residuals1(layer_34, spa_band, spa_band, 3, 5, 7)
    layer_36 = Inverted_residuals1(layer_35, spa_band, spa_band, 3, 5, 7)
    stage_4 = tf.add(layer_33, layer_36)

    layer_37 = Inverted_residuals1(stage_4, spa_band, spa_band, 3, 5, 7)
    stage_5 = layer_37

    fusion_spe = BN_layer(conv2d(stage_1, spe_band, 4 * spa_band, 1), is_training=is_training)
    fusion_spa1 = tf.add(stage_2, stage_3)
    fusion_spa2 = tf.add(fusion_spa1, stage_4)
    fusion_spa3 = tf.add(fusion_spa2, stage_5)
    fusion_spa3 = BN_layer(conv2d(fusion_spa3, spa_band, 4 * spa_band, 1), is_training=is_training)
    fusion_spe_spa = BN_layer(tf.add(fusion_spe, fusion_spa3), is_training=is_training)
    feature_spe_spa = avg_pool(fusion_spe_spa)
    W_fc_spe_spa = weight_variable([4 * spa_band, n_classes])
    b_fc_spe_spa = bias_variable([n_classes])
    feature_flat_spe_spa = tf.reshape(feature_spe_spa, [-1, 4 * spa_band])
    pred = tf.matmul(feature_flat_spe_spa, W_fc_spe_spa) + b_fc_spe_spa

    return pred
