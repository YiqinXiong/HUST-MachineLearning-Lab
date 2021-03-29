# -*- coding: utf-8 -*-
# Author: 熊逸钦
# Time: 2020/7/25 17:04


import math

import pandas as pd
from scipy.stats import norm


# 拆分训练集和测试集(7:3)
def from_csv_to_data_frame(csv_file):
    df = pd.read_csv(csv_file)
    # 分男女，对每一列求非空值的平均
    df_male = df[df['label'] == 'male'].iloc[:, 0:20]
    df_female = df[df['label'] == 'female'].iloc[:, 0:20]
    df_male_valid_mean = df_male[df_male != 0].mean()
    df_female_valid_mean = df_female[df_female != 0].mean()
    # 分性别，按照平均值填补空缺值
    for i in df_male.columns:
        df.loc[(df['label'] == 'male') & (df[i] == 0), i] = df_male_valid_mean[i]
        df.loc[(df['label'] == 'female') & (df[i] == 0), i] = df_female_valid_mean[i]
    # 打乱顺序后的data frame，先混洗，再随机抽取70%作为训练集，再将剩余的30%作为测试集
    df = df.sample(frac=1.0)
    df_train = df.sample(frac=0.7)
    df_test = df[~df.index.isin(df_train.index)]
    return df_train, df_test


# 获取属性列的平均值μ和标准差σ
def get_mean_std(df_train):
    # 计算每一个属性列的平均值和标准差
    df_train_male = df_train[df_train['label'] == 'male']
    df_train_female = df_train[df_train['label'] == 'female']
    mean_train_male = df_train_male.iloc[:, 0:20].mean().tolist()
    std_train_male = df_train_male.iloc[:, 0:20].std().tolist()
    mean_train_female = df_train_female.iloc[:, 0:20].mean().tolist()
    std_train_female = df_train_female.iloc[:, 0:20].std().tolist()
    return mean_train_male, std_train_male, mean_train_female, std_train_female


# 计算各个属性值的相对偏差，以相对偏差代表权重
def get_weight(mean_1, mean_2):
    weight = []
    # 计算各个属性值的相对偏差
    for i in range(len(mean_1)):
        a_male = mean_1[i]
        a_female = mean_2[i]
        weight.append(100 * abs(a_male - a_female) / ((a_male + a_female) / 2))
    # 将相对偏差规范化处理
    sum_weight = sum(weight)
    for i in range(len(weight)):
        weight[i] = (20 * weight[i]) / sum_weight
    return weight


# 朴素贝叶斯分类器
def naive_bayes_classifier(row_id, is_log=True, is_weight=True):
    if is_log:
        p_male_cond = math.log(p_male)
        p_female_cond = math.log(p_female)
    else:
        p_male_cond = p_male
        p_female_cond = p_female
    # 遍历每一属性列
    for i in range(test_df.shape[1] - 1):
        # 带权重时取属性权重，否则取1.0
        weight = attr_weight[i] if is_weight else 1.0
        # 用高斯分布函数计算条件概率
        g_male = norm.cdf(test_df.iloc[row_id, i], mean_male[i], std_male[i])
        g_female = norm.cdf(test_df.iloc[row_id, i], mean_female[i], std_female[i])
        # 取对数时计算条件概率的对数累加之和，否则计算条件概率累乘之积
        if is_log:
            p_male_cond += weight * math.log(g_male)
            p_female_cond += weight * math.log(g_female)
        else:
            p_male_cond *= pow(g_male, weight)
            p_female_cond *= pow(g_female, weight)
    return 'male' if p_male_cond > p_female_cond else 'female'


if __name__ == '__main__':
    # 从csv文件读取出训练集和测试集的data_frame
    train_df, test_df = from_csv_to_data_frame("voice.csv")
    # 得到P(男)和P(女)的先验概率
    p_male = len(train_df[train_df['label'] == 'male']) / len(train_df)
    p_female = 1 - p_male
    # 得到训练集数据中各列平均值和标准差，用于计算高斯分布概率
    mean_male, std_male, mean_female, std_female = get_mean_std(train_df)
    # 得到各个属性列的权重
    attr_weight = get_weight(mean_male, mean_female)
    # 最终输出的一些统计量
    male_all = len(test_df[test_df['label'] == 'male'])
    female_all = len(test_df[test_df['label'] == 'female'])
    male_hit = 0
    female_hit = 0
    # 对测试集中的每个成员，比较P(男|条件)和P(女|条件)的大小，取对数累加
    for row in range(len(test_df)):
        # # debug用
        # collect_male = []
        # collect_female = []

        # 使用贝叶斯分类器对样本进行分类
        result = naive_bayes_classifier(row, True, True)
        # 判断分类结果是否正确
        if test_df.iloc[row, test_df.shape[1] - 1] == result:
            if result == 'male':
                male_hit += 1
            else:
                female_hit += 1
            # print("[Y]", p_male_cond, p_female_cond, result, test_label[row][0])
        # else:
        # print("[N]", p_male_cond, p_female_cond, result, test_label[row][0])
        # print(collect_male)
        # print(collect_female)
    print(
        "男性测试数：%5d\t正确数：%5d\t正确率：%.4f" % (
            male_all,
            male_hit,
            male_hit / male_all))
    print(
        "女性测试数：%5d\t正确数：%5d\t正确率：%.4f" % (
            female_all,
            female_hit,
            female_hit / female_all))
    print(
        "测试集大小：%5d\t正确数：%5d\t正确率：%.4f" % (
            male_all + female_all,
            male_hit + female_hit,
            (male_hit + female_hit) / (male_all + female_all)))
