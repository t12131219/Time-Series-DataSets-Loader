"""
时间序列数据集加载工具集
包含UCR/UEA分类数据集、预测数据集、异常检测数据集的加载函数
"""

# 基础库导入
import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target  # 自定义工具函数
from scipy.io.arff import loadarff  # ARFF格式文件读取
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据标准化


def load_UCR(dataset):
    """
    加载UCR时间序列分类数据集

    参数:
        dataset (str): 数据集名称，对应datasets/UCR/下的子目录名

    返回:
        tuple: (训练数据, 训练标签, 测试数据, 测试标签)
        数据形状: (样本数, 序列长度, 1)
    """
    # 构建文件路径
    train_file = os.path.join('../datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('../datasets/UCR', dataset, dataset + "_TEST.tsv")

    # 读取TSV文件
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)

    # 转换为numpy数组
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # 标签编码处理（转换为0开始的连续整数）
    labels = np.unique(train_array[:, 0])
    transform = {l: i for i, l in enumerate(labels)}

    # 提取特征和标签
    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # 数据集特定处理：跳过已标准化的数据集
    no_normalize_list = [
        'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
        'BME', 'Chinatown', 'Crop', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian',
        'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure',
        'PigCVP', 'PLAID', 'PowerCons', 'Rock', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'SmoothSubspace', 'UMD'
    ]

    if dataset not in no_normalize_list:
        # 全局标准化（保持振幅信息的标准化方式）
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std

    # 增加通道维度
    return (train[..., np.newaxis], train_labels,
            test[..., np.newaxis], test_labels)


def load_UEA(dataset):
    """
    加载UEA多变量时间序列分类数据集

    参数:
        dataset (str): 数据集名称，对应datasets/UEA/下的子目录名

    返回:
        tuple: (训练数据, 训练标签, 测试数据, 测试标签)
        数据形状: (样本数, 特征维度, 序列长度)
    """
    # 读取ARFF格式文件
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        """辅助函数：从ARFF数据中提取特征和标签"""
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            # 转换数据格式
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")  # 字节转字符串
            res_data.append(t_data)
            res_labels.append(t_label)
        # 调整维度为(样本数, 特征数, 序列长度)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    # 处理训练测试数据
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    # 数据标准化（按特征维度标准化）
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    # 标签编码
    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    return train_X, train_y, test_X, test_y


def load_forecast_npy(name, univar=False):
    """
    加载.npy格式的预测数据集

    参数:
        name (str): 数据集名称（不含扩展名）
        univar (bool): 是否使用单变量模式

    返回:
        tuple: 包含数据切片、标准化器、预测长度等信息的元组
    """
    # 加载数据
    data = np.load(f'datasets/{name}.npy')

    # 单变量模式处理
    if univar:
        data = data[: -1:]  # 保留最后一列作为目标变量

    # 划分数据集（6:2:2比例）
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    # 标准化处理
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)  # 增加批次维度

    # 预定义的预测长度
    pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    """
    生成时间相关特征（辅助函数）

    参数:
        dt (DatetimeIndex): 时间索引

    返回:
        np.ndarray: 时间特征矩阵，形状为(时间步数, 7)
        特征包括：分钟、小时、星期几、月中日、年中日、月份、年中周数
    """
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    """
    加载CSV格式的时序预测数据集

    参数:
        name (str): 数据集名称（不含扩展名）
        univar (bool): 是否使用单变量模式

    返回:
        tuple: 包含数据切片、标准化器、预测长度等信息的元组
    """
    # 读取CSV文件
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)

    # 生成时间特征
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]  # 时间特征维度

    # 单变量模式处理
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]  # 特定数据集的目标列
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]  # 默认取最后一列

    # 转换为numpy数组
    data = data.to_numpy()

    # 数据集特定的切片划分
    if name in ('ETTh1', 'ETTh2'):
        # ETTh数据集：20个月数据，按12/4/4划分
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name in ('ETTm1', 'ETTm2'):
        # ETTm数据集：数据频率更高（15分钟间隔）
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    else:
        # 默认6:2:2划分
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    # 数据标准化
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    # 数据维度调整
    if name == 'electricity':
        data = np.expand_dims(data.T, -1)  # 电力数据集特殊处理（变量作为实例）
    else:
        data = np.expand_dims(data, 0)  # 增加批次维度

    # 合并时间特征
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    # 设置预测长度
    pred_lens = [24, 48, 168, 336, 720] if name in ('ETTh1', 'ETTh2', 'electricity') else [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    """
    加载异常检测数据集

    参数:
        name (str): 数据集名称（不含扩展名）

    返回:
        tuple: 包含训练数据、标签、时间戳和异常延迟信息
    """
    res = pkl_load(f'datasets/{name}.pkl')  # 加载pickle文件
    return (res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'],
            res['all_test_data'], res['all_test_labels'], res['all_test_timestamps'],
            res['delay'])


def gen_ano_train_data(all_train_data):
    """
    生成异常检测训练数据（带填充）

    参数:
        all_train_data (dict): 原始训练数据字典

    返回:
        np.ndarray: 填充后的训练数据，形状为(样本数, 最大序列长度, 1)
    """
    # 计算最大序列长度
    maxl = np.max([len(all_train_data[k]) for k in all_train_data])

    # 填充并堆叠数据
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)

    return np.expand_dims(np.stack(pretrain_data), 2)