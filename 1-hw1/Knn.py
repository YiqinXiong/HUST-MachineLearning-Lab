# -*- coding: utf-8 -*-
# Author: 熊逸钦
# Time: 2020/5/10 16:55

import operator
import struct

import matplotlib.pyplot as plt
import numpy as np


# 读取idx3文件为手写数字图片
def idx3_to_images(idx3_filename):
    # 读取二进制数据
    bin_data = open(idx3_filename, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(
            struct.unpack_from(
                fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


# 读取idx1文件为标签
def idx1_to_labels(idx1_filename):
    # 读取二进制数据
    bin_data = open(idx1_filename, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 标签数量: %d个' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '个标签')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# K取最佳值3之后进行测试，10000个测试点
def test_k_3():
    # 训练集文件
    train_images_filename = 'train-images.idx3-ubyte'
    # 训练集标签文件
    train_labels_filename = 'train-labels.idx1-ubyte'
    # 测试集文件
    test_images_filename = 't10k-images.idx3-ubyte'
    # 测试集标签文件
    test_labels_filename = 't10k-labels.idx1-ubyte'

    # 导入训练集
    train_images = idx3_to_images(train_images_filename)
    train_labels = idx1_to_labels(train_labels_filename)
    # 导入测试集
    test_images = idx3_to_images(test_images_filename)
    test_labels = idx1_to_labels(test_labels_filename)
    # 将(60000,28,28)整理为(60000,28*28)
    train_size = train_images.shape[0]  # 样本个数
    test_size = test_images.shape[0]  # 样本个数
    train_images = np.reshape(train_images, (train_size, 28 * 28))
    test_images = np.reshape(test_images, (test_size, 28 * 28))

    # 参数设置，包括k的范围设置以及测试的数据集大小设置
    k = 3
    conf_test_begin = 0
    conf_test_size = 10000
    # 错误计数
    err_count = 0.0
    # 对测试集中的各个数据进行测试
    for i in range(conf_test_begin, conf_test_size):
        classifier_result, neighbors = knn(test_images[i], train_images, train_labels, k)
        print('测试点%5d|\t(预测:%d 答案:%d)' % (i + 1, classifier_result, test_labels[i]), end='\t')
        print('K近邻:', end='')
        print(neighbors)
        if classifier_result != test_labels[i]:
            err_count += 1.0
    # 计算误差率
    err_rate = err_count / float(conf_test_size)
    print('K=%d err_count:%d err_rate:%f' % (k, err_count, err_rate))
    return


# 对测试集进行测试，1000*20个测试点
def test():
    # 训练集文件
    train_images_filename = 'train-images.idx3-ubyte'
    # 训练集标签文件
    train_labels_filename = 'train-labels.idx1-ubyte'
    # 测试集文件
    test_images_filename = 't10k-images.idx3-ubyte'
    # 测试集标签文件
    test_labels_filename = 't10k-labels.idx1-ubyte'

    # 导入训练集
    train_images = idx3_to_images(train_images_filename)
    train_labels = idx1_to_labels(train_labels_filename)
    # 导入测试集
    test_images = idx3_to_images(test_images_filename)
    test_labels = idx1_to_labels(test_labels_filename)
    # 将(60000,28,28)整理为(60000,28*28)
    train_size = train_images.shape[0]  # 样本个数
    test_size = test_images.shape[0]  # 样本个数
    train_images = np.reshape(train_images, (train_size, 28 * 28))
    test_images = np.reshape(test_images, (test_size, 28 * 28))

    # 参数设置，包括k的范围设置以及测试的数据集大小设置
    conf_k_range = range(1, 21)
    conf_test_begin = 0
    conf_test_size = 1000
    # 记录绘图的横纵坐标
    mis_classification_rates = []
    # 下面对各个k都进行一次测试
    for k in conf_k_range:
        # 错误计数
        err_count = 0.0
        # 对测试集中的各个数据进行测试
        for i in range(conf_test_begin, conf_test_size):
            classifier_result, neighbors = knn(test_images[i], train_images, train_labels, k)
            print('测试点%5d|\t(预测:%d 答案:%d)' % (i + 1, classifier_result, test_labels[i]), end='\t')
            print('K近邻:', end='')
            print(neighbors)
            if classifier_result != test_labels[i]:
                err_count += 1.0
        # 计算误差率
        err_rate = err_count / float(conf_test_size)
        print('K=%d err_count:%d err_rate:%f' % (k, err_count, err_rate))
        mis_classification_rates.append(err_rate)
    # 绘图
    plt.plot(conf_k_range, mis_classification_rates)
    plt.show()
    # 取使得误差率最小的k值
    print(mis_classification_rates)
    print('选择K=%d' % conf_k_range[mis_classification_rates.index(min(mis_classification_rates))])
    return


# 对训练集进行测试，1000*20个测试点
def train():
    # 训练集文件
    train_images_filename = 'train-images.idx3-ubyte'
    # 训练集标签文件
    train_labels_filename = 'train-labels.idx1-ubyte'

    # 导入训练集
    train_images = idx3_to_images(train_images_filename)
    train_labels = idx1_to_labels(train_labels_filename)
    # 将(60000,28,28)整理为(60000,28*28)
    train_size = train_images.shape[0]  # 样本个数
    train_images = np.reshape(train_images, (train_size, 28 * 28))

    # 参数设置，包括k的范围设置以及测试的数据集大小设置
    conf_k_range = range(1, 21)
    conf_test_begin = 0
    conf_test_size = 1000
    # 记录绘图的横纵坐标
    mis_classification_rates = []
    # 下面对各个k都进行一次测试
    for k in conf_k_range:
        # 错误计数
        err_count = 0.0
        # 对测试集中的各个数据进行测试
        for i in range(conf_test_begin, conf_test_size):
            classifier_result, neighbors = knn(train_images[i], train_images, train_labels, k)
            print('测试点%5d|\t(预测:%d 答案:%d)' % (i + 1, classifier_result, train_labels[i]), end='\t')
            print('K近邻:', end='')
            print(neighbors)
            if classifier_result != train_labels[i]:
                err_count += 1.0
        # 计算误差率
        err_rate = err_count / float(conf_test_size)
        print('K=%d err_count:%d err_rate:%f' % (k, err_count, err_rate))
        mis_classification_rates.append(err_rate)
    # 绘图
    plt.plot(conf_k_range, mis_classification_rates)
    plt.show()
    # 取使得误差率最小的k值
    print(mis_classification_rates)
    print('选择K=%d' % conf_k_range[mis_classification_rates.index(min(mis_classification_rates))])
    return


def knn(in_x, data_set, labels, k):
    # 训练样本个数
    data_set_size = data_set.shape[0]
    # tile让in_x重复data_set_size次，再与data_set相减得到diff_mat
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 将diff_mat平方后横向求和得到sq_distances（即Σ(ai-bi)^2），再开方得到in_x和data_set的各数据点的欧氏距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 将距离从小到大排序，排序结果为数据集中数据点的序号形成的列表
    sorted_dist = distances.argsort()
    # 字典存储不同标签出现的次数
    class_count = {}
    # 记录k个邻居
    neighbors = []
    for i in range(k):
        # 获取当前标签并加入到邻居列表中
        cur_label = labels[sorted_dist[i]]
        neighbors.append(int(cur_label))
        # 统计标签出现次数
        class_count[cur_label] = class_count.get(cur_label, 0) + 1
    # 按‘出现次数’对标签字典进行排序，出现次数多的排前面，operator.itemgetter选择‘出现次数’为排序依据
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0], neighbors


if __name__ == '__main__':
    test_k_3()
