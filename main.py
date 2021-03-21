# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from spasa import mat_data, data_PU, mat_gt, gt_PU, Height, Width, Band, PATCH_LENGTH
from spasa import overall_sample_divide
import moudle
import os
import scipy.io
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Parameters
model_num = 10 # 不同训练样本迭代次数
model_iter = 1 # 相同训练样本迭代次数
patch_size = PATCH_LENGTH * 2 + 1
n_input = Band * patch_size * patch_size

n_classes = 9
batch_size = 32
training_iters = 4001
learning_rate = 0.001
XDNN = np.zeros(shape=(model_num * model_iter, n_classes + 5))

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
is_training = tf.placeholder(dtype=tf.float32)
n = tf.placeholder(dtype=tf.float32)

# Construct model
num_band = Band
net = tf.reshape(x, [-1, patch_size, patch_size, num_band])

def count(graph):
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def stats_graph(graph):
    profiler = tf.profiler.Profiler(graph=sess.graph)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    float_stats = profiler.profile_operations(opts)
    return float_stats.total_float_ops

def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0+= dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    zsum = np.sum(ysum, axis=1)
    # xsum是个k行1列的向量，ysum是个1行k列的向量
    Pe  = float(ysum*xsum)/zsum**2
    P0 = float(P0/zsum*1.0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))
    return cohens_coefficient, zsum

pred = moudle.model_1(net)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)  # 交叉熵函数
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # 优化器

# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
predict_label = tf.argmax(pred, 1)

# Initializing the variables初始化变量
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        total_parameters = count(moudle.model_1)
        print(total_parameters)
        flops = stats_graph(moudle.model_1)
        print(flops)

        for exiter in range(model_num):
            TRAIN_SIZE, Training_Patch, Verifying_Patch, Test_Patch, Training_Label, Verifying_Label, \
            Test_Label, Data_Padding, TOTAL_SIZE, train_indices, val_indices, test_indices \
            = overall_sample_divide(data_PU, gt_PU, Band, PATCH_LENGTH)
            Training_Patch = np.reshape(Training_Patch, (-1, n_input))
            Verifying_Patch = np.reshape(Verifying_Patch, (-1, n_input))
            Test_Patch = np.reshape(Test_Patch, (-1, n_input))
            x_test, y_test = Test_Patch, Test_Label
            x_verify, y_verify = Verifying_Patch, Verifying_Label
            y_verify_scalar = np.argmax(y_verify, 1) + 1
            y_test_scalar = np.argmax(y_test, 1) + 1
            x_train, y_train = Training_Patch, Training_Label
            test_batch = 50000

            for m in range(model_iter):
                sess.run(init)
                true_label = []
                correct_num = 0
                accuracy_sum = 0
                all_map_label = np.zeros(shape=(Height, Width))
                highest_accuracy = 0
                start_time = time.time()
                for iteration in range(training_iters):
                    idx = np.random.choice(TRAIN_SIZE, size=batch_size, replace=False)
                    batch_x = x_train[idx, :]
                    batch_y = y_train[idx, :]
                    _, batch_cost, train_acc = sess.run([optimizer, cost, accuracy],
                        feed_dict={x:batch_x, y:batch_y, is_training:1, n:batch_x.shape[0]})  # feed_dict：赋值
                    if iteration % 10 == 0:
                        print("Iteraion = %04d," % (iteration), \
                              "Batch cost = %.4f," % (batch_cost), \
                              "Training Accuracy = %.4f" % (train_acc))
                    if iteration % 20 == 0:
                        print('Training Data Eval: Training Accuracy = %.4f' % sess.run(accuracy,
                                        feed_dict={x: x_train, y: y_train, is_training: 1, n:x_train.shape[0]}))
                        verify_acc = sess.run(accuracy,
                                        feed_dict={x: x_verify, y: y_verify, is_training: 0, n:x_verify.shape[0]})
                        if verify_acc >= highest_accuracy:
                            highest_accuracy = verify_acc
                            save_path = saver.save(sess, "F:\\NAS-LAFFN\\UP\\model\\UP-model.ckpt")
                            i = iteration + 1
                        print('Verify Data Eval: Verify Accuracy = %.4f' % verify_acc)
                print("Optimization Finished!")
                train_time = time.time() - start_time
                print("=======================================================================")
                print('Net training is Completed! (It takes %.5f seconds)' % (train_time))

                # Test model
                saver.restore(sess, "F:\\NAS-LAFFN\\UP\\model\\UP-model.ckpt")
                test_num = len(x_test)
                test_times = int(test_num / test_batch) + 1
                acc_num = 0
                acc_nums = 0
                pred_label = 0
                label = []
                test_label = []
                test_labels = []
                test_start = time.time()
                for i in range(test_times):
                    if i < test_times - 1:
                        x_temp = x_test[i * test_batch:(i + 1) * test_batch, :]
                        y_temp = y_test[i * test_batch:(i + 1) * test_batch, :]
                        acc_num, pred_label = sess.run([accuracy, pred],
                            feed_dict={x : x_temp, y : y_temp, is_training : 0, n : x_temp.shape[0]})
                        acc_nums += acc_num * len(x_temp)
                        test_labels.extend(pred_label)
                    else:
                        x_temp = x_test[i * test_batch:, :]
                        y_temp = y_test[i * test_batch:, :]
                        acc_num, pred_label = sess.run([accuracy, pred],
                            feed_dict={x : x_temp, y : y_temp, is_training : 0, n : x_temp.shape[0]})
                        acc_nums += acc_num * len(x_temp)
                        test_labels.extend(pred_label)
                test_time = time.time() - test_start
                test_acc = acc_nums / test_num
                print("The Final Test Accuracy is :", test_acc)
                print("=======================================================================")
                print('Net testing is Completed! (It takes %.5f seconds)' % (test_time))
                test_labels = np.argmax(test_labels, 1)
                true_label = np.argmax(y_test, 1)
                test_labels = np.reshape(test_labels, (-1, len(test_labels)))
                true_label = np.reshape(true_label, (-1, len(true_label)))
                chart = np.zeros(shape=(n_classes, n_classes))
                for j in range(len(y_test)):
                    x_sum = int(true_label[0][j])
                    y_sum = int(test_labels[0][j])
                    chart[x_sum, y_sum] += 1
                for k in range(n_classes):
                    accuracy_sum = accuracy_sum + chart[k, k]
                AA_mean = 0
                AA = np.zeros(n_classes)
                sum_1 = chart.sum(axis=1)
                for i in range(0, n_classes):
                    num = sum_1[i]
                    AA[i] = chart[i, i] / num
                    AA_mean = AA_mean + AA[i]
                    XDNN[exiter * model_iter + m, i] = float(AA[i])
                AA_mean = float(AA_mean / n_classes)
                accuracy_sum = float(accuracy_sum / test_num)
                K, sample_sum = kappa(chart, n_classes)
                print("")
                XDNN[exiter * model_iter + m, 9] = accuracy_sum
                XDNN[exiter * model_iter + m, 10] = AA_mean
                XDNN[exiter * model_iter + m, 11] = K
                XDNN[exiter * model_iter + m, 12] = train_time
                XDNN[exiter * model_iter + m, 13] = test_time
                print(exiter * model_iter + m)

                # 全图预测
                All_map_start = time.time()
                for i in range(0, Height):
                    for j in range(0, 1):
                        jm = Width * j
                        input_tem = np.zeros(shape=(Width, n_input))
                        for k in range(Width):
                            input_t = Data_Padding[i: i + patch_size, jm + k:jm + k + patch_size, :]
                            input_t = np.reshape(input_t, (-1, n_input))
                            input_tem[k, :] = input_t
                        pred_num = sess.run(pred, feed_dict={x: input_tem, is_training: 0, n: input_tem.shape[0]})
                        label = np.argmax(pred_num, 1)
                        label = np.reshape(label, (-1, Width))
                        all_map_label[i, :] = label
                All_map_time = time.time() - All_map_start
                print('All map is Completed! (It takes %.5f seconds)' % (All_map_time))
                print("")
                DATA_PATH = 'F:\\NAS-LAFFN\\UP\\dataset'
                Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'PaviaU_gt.mat'))['paviaU_gt']
                pre_map = np.zeros((Height, Width))
                pre_map = all_map_label
                dataNew = 'F:\\NAS-LAFFN\\UP\\map\\UP_' + str(exiter * model_iter + m) + 'map.mat'
                scipy.io.savemat(dataNew, {'UP': np.matrix(pre_map)})
        for i in range(model_num * model_iter):
            for j in range(n_classes + 5):
                print(XDNN[i][j])
            print("")
        Average_XDNN = XDNN.sum(axis=0) / (model_num * model_iter)
        print(Average_XDNN)
