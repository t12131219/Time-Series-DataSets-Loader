import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import numpy as np
from scipy.io.arff import loadarff  # ARFF格式文件读取
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据标准化
from statsmodels.tsa.stattools import acf, pacf
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

'''
parameters['Save_OR_Not'] = True    # 是否保存结果图
parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
parameters['cut_length'] = 30       # 截断长度
parameters['CDF_OR_Not'] = False    # 是否显示CDF图
'''


def remove_after_dot(s):
    """
    去除字符串中第一个 '.' 及其后面的所有字符。

    参数:
        s (str): 输入字符串。

    返回:
        str: 去除 '.' 后面部分的字符串。
    """
    if '.' in s:
        return s.split('.')[0],s.split('.')[1]
    else:
        return s

def load_UCR(dataset):
    # 构建文件路径
    train_file = os.path.join('../datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('../datasets/UCR', dataset, dataset + "_TEST.tsv")

    # 读取TSV文件
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    return train_df, test_df

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

def read_tsf_file(file_path, data_encoding='latin-1'):
    """
    二进制模式读取TSF文件，自动处理编码问题，并返回与read_csv格式一致的pandas.DataFrame

    参数:
    file_path (str): 文件路径
    data_encoding (str): 数据部分解码编码，默认为latin-1

    返回:
    pandas.DataFrame: 包含分割数据的DataFrame
    """
    dataset = []
    in_data = False

    with open(file_path, 'rb') as file:  # 二进制模式打开
        for byte_line in file:
            # 清理首尾空白字节
            stripped = byte_line.strip()

            if not in_data:
                # 检测数据段开始标记（不区分大小写）
                if stripped.lower() == b'@data':
                    in_data = True
                continue

            # 处理数据段内容
            if stripped:
                try:
                    # 解码并转换为字符串
                    str_line = stripped.decode(data_encoding)
                except UnicodeDecodeError:
                    # 解码失败时使用替换策略
                    str_line = stripped.decode(data_encoding, errors='replace')

                # 将逗号和冒号都作为分隔符
                str_line = str_line.replace(':', ',')
                # 分割数据并清理元素
                row = [item.strip() for item in str_line.split(',')]
                dataset.append(row)

    # 将二维列表转换为pandas DataFrame
    df = pd.DataFrame(dataset)

    # 剔除第一列
    if not df.empty:
        # 提取第一列
        first_column = df.iloc[:, 1]
        # 剔除第一列
        df = df.iloc[:, 2:]
        # 将剔除后的第一列的标题变为'date'
        df.insert(0, 'date', first_column)

    return df
#%%
def Draw_Visiual_Single_Feature(df, columns_list, parameters, save_dir):
    '''
    parameters['Save_OR_Not'] = True    # 是否保存结果图
    parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
    parameters['cut_length'] = 30       # 截断长度
    parameters['CDF_OR_Not'] = False    # 是否显示CDF图
    parameters['file_name'] = file_name
    '''
    df = df.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    # #     df_filtered = df.loc["2014-12-01":"2014-12-14"]
    # #     df = df_filtered
    if 'date' not in columns_list and 'Date' not in columns_list:
        print("Date not in columns_list")
        # 创建多子图布局
        num_features = len(columns_list)
        fig, axes = plt.subplots(num_features, 1, figsize=(15, 10 * num_features))  # figsize= width * height
        colors = ['blue', 'red', 'green', 'orange','yellow', 'pink']
        # 逐行绘制每个特征
        for i, feature in enumerate(df.columns):
            # print(feature)
            ax = axes[i]
            axes[i].plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)])
            # axes[i].plot(df.index, df[feature], marker='o', linestyle='-', color='blue')
            axes[i].set_title(f'Feature: {parameters["file_name"]}_{feature}')
            axes[i].set_xlabel('Row Index')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)

        plt.tight_layout()  # 自动调整子图间距
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Single_Feature')
        plt.show()

        #  # 在同一张图中绘制所有特征
        # plt.figure(figsize=(15, 10))
        # for i, feature in enumerate(df.columns):
        #     plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)
        #
        # # 设置标题、标签和图例
        # plt.title('All Features in One Plot')
        # plt.xlabel('Row Index')
        # plt.ylabel('Value')
        # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放在右侧外部
        # plt.grid(True)
        # plt.tight_layout()  # 自动调整布局
        # plt.show()

    elif 'date' in columns_list:
        print("date in columns_list")
        # print(df['date'])
        # num_features = len(df.columns)-1
        # fig, axes = plt.subplots(num_features, 1, figsize=(20, 6 * num_features))
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df.set_index('date').plot(subplots=True, figsize=(15, 40))
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Single_Feature')
        plt.show()
    elif 'Date' in columns_list:
        # num_features = len(df.columns)-1
        # fig, axes = plt.subplots(num_features, 1, figsize=(20, 6 * num_features))
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date').plot(subplots=True, figsize=(15, 40))
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Single_Feature')
        plt.show()

def Draw_Visiual_Single_Feature_Each_Picture(df, columns_list, parameters, save_dir):
    '''
    parameters['Save_OR_Not'] = True    # 是否保存结果图
    parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
    parameters['cut_length'] = 30       # 截断长度
    parameters['CDF_OR_Not'] = False    # 是否显示CDF图
    parameters['file_name'] = file_name
    '''
    df = df.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]

    i=0
    colors = ['blue', 'red', 'green', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'brown']
    if 'date' not in columns_list and 'Date' not in columns_list:
        print("Date not in columns_list")

        # 为每个特征单独绘制并保存图片
        for feature in columns_list:
            if feature == 'Index': pass
            plt.figure(figsize=(15, 6))  # 设置图形大小
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)
            plt.title(f'Feature: {feature}', fontsize=14)
            plt.xlabel('Row Index', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True)
            if parameters['Save_OR_Not']:
                # 保存图片
                save_path = f"{save_dir}/{parameters['file_name']}_{feature}.png"
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"目录 {directory} 已创建。")
                plt.savefig(save_path)
                print(f"Saved: {save_path}")
            i=i+1
            plt.close()  # 关闭当前图形，避免内存占用过高

    elif 'date' in columns_list:
        print("date in columns_list")
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df.set_index('date', inplace=True)

        # 为每个特征单独绘制并保存图片
        for feature in columns_list:
            if feature == 'date':
                continue  # 跳过日期列
            plt.figure(figsize=(15, 6))
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)
            plt.title(f'Feature: {feature}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True)

            if parameters['Save_OR_Not']:
                # 保存图片
                save_path = f"{save_dir}/{parameters['file_name']}_{feature}.png"
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"目录 {directory} 已创建。")
                plt.savefig(save_path)
                print(f"Saved: {save_path}")
            i=i+1
            plt.close()

    elif 'Date' in columns_list:
        print("Date in columns_list")
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)

        # 为每个特征单独绘制并保存图片
        for feature in columns_list:
            if feature == 'Date':
                continue  # 跳过日期列
            plt.figure(figsize=(15, 6))
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)
            plt.title(f'Feature: {feature}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True)

            if parameters['Save_OR_Not']:
                # 保存图片
                save_path = f"{save_dir}/{parameters['file_name']}_{feature}.png"
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"目录 {directory} 已创建。")
                plt.savefig(save_path)
                print(f"Saved: {save_path}")
            i=i+1
            plt.close()

    elif 'timestamp' in columns_list:
        print("Date in columns_list")
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)

        # 为每个特征单独绘制并保存图片
        for feature in columns_list:
            if feature == 'Date':
                continue  # 跳过日期列
            plt.figure(figsize=(15, 6))
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)
            plt.title(f'Feature: {feature}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True)

            if parameters['Save_OR_Not']:
                # 保存图片
                save_path = f"{save_dir}/{parameters['file_name']}_{feature}.png"
                directory = os.path.dirname(save_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"目录 {directory} 已创建。")
                plt.savefig(save_path)
                print(f"Saved: {save_path}")
            i=i+1
            plt.close()

#%%
def Draw_Visiual_Multi_Feature(df, columns_list, parameters, save_dir):
    '''
    parameters['Save_OR_Not'] = True    # 是否保存结果图
    parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
    parameters['cut_length'] = 30       # 截断长度
    parameters['CDF_OR_Not'] = False    # 是否显示CDF图
    parameters['file_name'] = file_name
    '''
    df = df.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    # 检查是否包含日期列
    if 'date' not in columns_list and 'Date' not in columns_list:
        # 创建一张图
        plt.figure(figsize=(15, 10))  # 设置图像大小
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink', 'purple', 'brown', 'gray', 'cyan']

        # 在同一张图中绘制所有特征
        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)

        # 设置标题、标签和图例
        plt.title('All Features in One Plot')
        plt.xlabel('Row Index')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放在右侧外部
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature')
        plt.show()

    # 如果包含日期列
    elif 'date' in columns_list:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df.set_index('date', inplace=True)
        plt.figure(figsize=(15, 10))  # 设置图像大小
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink', 'purple', 'brown', 'gray', 'cyan']

        # 在同一张图中绘制所有特征
        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)

        # 设置标题、标签和图例
        plt.title('All Features in One Plot')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放在右侧外部
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature')
        plt.show()

    elif 'Date' in columns_list:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)
        plt.figure(figsize=(15, 10))  # 设置图像大小
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink', 'purple', 'brown', 'gray', 'cyan']

        # 在同一张图中绘制所有特征
        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)

        # 设置标题、标签和图例
        plt.title('All Features in One Plot')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放在右侧外部
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature')
        plt.show()

    elif 'timestamp' in columns_list:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)
        plt.figure(figsize=(15, 10))  # 设置图像大小
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink', 'purple', 'brown', 'gray', 'cyan']

        # 在同一张图中绘制所有特征
        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-', color=colors[i % len(colors)], label=feature)

        # 设置标题、标签和图例
        plt.title('All Features in One Plot')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放在右侧外部
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature')
        plt.show()

#%%
def Draw_Visiual_Multi_Feature_With_ZNormalization(df, columns_list, parameters, save_dir):
    '''
    parameters['Save_OR_Not'] = True    # 是否保存结果图
    parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
    parameters['cut_length'] = 30       # 截断长度
    parameters['CDF_OR_Not'] = False    # 是否显示CDF图
    parameters['file_name'] = file_name
    '''

    df = df.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]


    # 定义归一化函数
    def z_normalize(data):
        return (data - data.mean()) / data.std()

    # 检查是否包含日期列
    if 'date' not in columns_list and 'Date' not in columns_list:
        # 执行归一化
        df[df.columns] = z_normalize(df)

        # 创建一张图
        plt.figure(figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink',
                 'purple', 'brown', 'gray', 'cyan']

        # 绘制所有特征
        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-',
                    color=colors[i % len(colors)], label=feature)

        # 设置图表属性
        plt.title('Normalized Features Visualization')
        plt.xlabel('Row Index')
        plt.ylabel('Z-Score Normalized Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature_With_ZNormalization')
        plt.show()

    elif 'date' in columns_list:
        # 处理日期列
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 执行归一化
        df[df.columns] = z_normalize(df)

        # 创建可视化
        plt.figure(figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink',
                 'purple', 'brown', 'gray', 'cyan']

        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-',
                    color=colors[i % len(colors)], label=feature)

        plt.title('Normalized Temporal Features')
        plt.xlabel('Date')
        plt.ylabel('Z-Score Normalized Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature_With_ZNormalization')
        plt.show()

    elif 'Date' in columns_list:
        # 处理日期列
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 执行归一化
        df[df.columns] = z_normalize(df)

        # 创建可视化
        plt.figure(figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'orange', 'yellow', 'pink',
                 'purple', 'brown', 'gray', 'cyan']

        for i, feature in enumerate(df.columns):
            plt.plot(df.index, df[feature], linestyle='-',
                    color=colors[i % len(colors)], label=feature)

        plt.title('Normalized Temporal Features')
        plt.xlabel('Date')
        plt.ylabel('Z-Score Normalized Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        Save_Fig(parameters['Save_OR_Not'], save_dir, 'Draw_Visiual_Multi_Feature_With_ZNormalization')
        plt.show()
#%%
def plot_features_with_selector(df, columns_list, flag, save_dir):
    df = df.copy()
    # 检查是否包含日期列
    if 'date' not in columns_list and 'Date' not in columns_list:
        # 将数据转换为长格式
        df_long = df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
        x_col = 'index'
        x_label = 'Row Index'

    elif 'date' in columns_list:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df.set_index('date', inplace=True)
        df_long = df.reset_index().melt(id_vars='date', var_name='Feature', value_name='Value')
        x_col = 'date'
        x_label = 'Date'

    elif 'Date' in columns_list:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)
        df_long = df.reset_index().melt(id_vars='Date', var_name='Feature', value_name='Value')
        x_col = 'Date'
        x_label = 'Date'

    # 创建基础图表
    fig = px.line(df_long, x=x_col, y='Value', color='Feature',
                  title='Feature Visualization with Selector',
                  labels={x_col: x_label, 'Value': 'Value'})

    # 添加下拉菜单
    buttons = []
    features = df_long['Feature'].unique()

    # 全选按钮
    buttons.append(dict(
        label='All Features',
        method='update',
        args=[{"visible": [True]*len(features)},  # 显示所有特征
              {"title": "All Features"}]
    ))

    # 单个特征按钮
    for i, feature in enumerate(features):
        visible = [f == feature for f in features]  # 仅当前特征可见
        buttons.append(dict(
            label=feature,
            method='update',
            args=[{"visible": visible},  # 更新可见性
                  {"title": f"Feature: {feature}"}]  # 更新标题
        ))

    # 添加范围选择器（保持你原来的时间选择功能）
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # 添加下拉菜单到布局
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.1,
            y=1.15,
            buttons=buttons
        )]
    )
    Save_Fig(flag, save_dir, 'plot_features_with_selector')
    fig.show()
#%%
def plot_features_horizontal_selector(df, columns_list, flag, save_dir):
    df = df.copy()
    # 检查是否包含日期列
    if 'date' not in columns_list and 'Date' not in columns_list:
        df_long = df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
        x_col = 'index'
        x_label = 'Row Index'
    elif 'date' in columns_list:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df.set_index('date', inplace=True)
        df_long = df.reset_index().melt(id_vars='date', var_name='Feature', value_name='Value')
        x_col = 'date'
        x_label = 'Date'
    elif 'Date' in columns_list:
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        df.set_index('Date', inplace=True)
        df_long = df.reset_index().melt(id_vars='Date', var_name='Feature', value_name='Value')
        x_col = 'Date'
        x_label = 'Date'

    # 创建基础图表
    fig = px.line(df_long, x=x_col, y='Value', color='Feature',
                  title='<b>Interactive Feature Selection</b>',
                  labels={x_col: x_label, 'Value': 'Value'})

    # 生成按钮配置
    features = df_long['Feature'].unique().tolist()
    buttons = [
        dict(
            label='ALL',
            method='update',
            args=[{"visible": [True]*len(features)}]
        )
    ]
    buttons += [
        dict(
            label=feature,
            method='update',
            args=[{"visible": [f == feature for f in features]}]
        ) for feature in features
    ]

    # 横向按钮布局参数
    button_layout = dict(
        type="buttons",
        direction="right",    # 横向排列
        showactive=True,
        x=0.5,              # 水平居中
        xanchor="center",    # 锚点居中
        y=1.15,             # 位于图表上方
        yanchor="top",
        pad={"r": 10, "t": 10},
        buttons=buttons
    )

    # 时间范围选择器布局调整
    range_selector = dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(step="all")
        ]),
        y=0.95,  # 下移避免重叠
        xanchor="left"
    )

    # 更新图表布局
    fig.update_layout(
        updatemenus=[button_layout],
        xaxis_rangeslider_visible=True,
        xaxis_rangeselector=range_selector,
        margin=dict(t=150),  # 增加顶部边距
        plot_bgcolor='rgba(240,240,240,0.8)',
        height=600  # 固定高度
    )

    # 优化显示效果
    fig.update_traces(
        hovertemplate="<br>".join([
            f"{x_label}: %{{x}}",
            "Value: %{y:.2f}",
        ])
    )
    Save_Fig(flag, save_dir, 'plot_features_horizontal_selector')
    fig.show()

def Save_Fig(flag, save_dir, save_name):
    if not flag:
       return
    filename = save_dir+'/'+ save_name
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")

    # 保存图形为图片文件
    plt.savefig(filename)
    print(f"图形已保存为: {filename}")
    return

# -------------------------------------------------------------------------------------------------------

def CDF_Visual(data, columns_list, parameters):
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]


    columns_list # 替换为你想要绘制CDF的列名
    for i, feature in enumerate(data.columns):
        # 计算累积分布函数
        cdf_values = np.sort(data[feature])  # 对数据进行排序
        cdf_values2 = np.arange(1, len(cdf_values) + 1) / len(cdf_values)

        # 绘制CDF图
        plt.figure(figsize=(10, 6))
        plt.step(cdf_values, cdf_values2, where='post', label=feature)
        plt.title(f'CDF of {feature}')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)
        plt.show()

def CDF_Visual_Save(data, columns_list, parameters, save_dir):
    data = data.copy()
    if len(columns_list) > parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ", parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]

    # 替换为你想要绘制CDF的列名
    for i, feature in enumerate(columns_list):
        if feature == 'date':
            continue
        # 计算累积分布函数
        cdf_values = np.sort(data[feature])  # 对数据进行排序
        cdf_values2 = np.arange(1, len(cdf_values) + 1) / len(cdf_values)

        # 绘制CDF图
        plt.figure(figsize=(10, 6))
        plt.step(cdf_values, cdf_values2, where='post', label=feature)
        plt.title(f'CDF of {feature}')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)

        # 保存图像
        output_dir = save_dir + "/CDF_Plots"  # 定义保存图像的目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # 如果目录不存在，则创建
        output_file = os.path.join(output_dir, f"{feature}_CDF.png")
        plt.savefig(output_file)  # 保存图像
        print(f"Saved CDF plot for {feature} to {output_file}")
        # 关闭当前图像（避免占用过多内存）
        plt.close()


def PDF_Visual(data, columns_list, parameters, bandwidth=0.5):
    """概率密度函数可视化"""
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    for feature in columns_list:
        # 计算核密度估计
        kde = gaussian_kde(data[feature].dropna(), bw_method=bandwidth)
        x = np.linspace(data[feature].min(), data[feature].max(), 1000)

        plt.figure(figsize=(10, 6))
        plt.plot(x, kde(x), label='KDE', lw=2)
        plt.title(f'PDF of {feature} (bandwidth={bandwidth})')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.legend()
        plt.show()


def ACF_Visual(data, columns_list, parameters, lags=40):
    """自相关函数可视化"""
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    for feature in columns_list:
        plt.figure(figsize=(12, 6))
        plot_acf(data[feature].dropna(), lags=lags, title=f'ACF of {feature}')
        plt.grid(True)
        plt.show()


def PACF_Visual(data, columns_list, parameters, lags=40):
    """偏自相关函数可视化"""
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    for feature in columns_list:
        plt.figure(figsize=(12, 6))
        plot_pacf(data[feature].dropna(), lags=lags, title=f'PACF of {feature}')
        plt.grid(True)
        plt.show()


def PSD_Visual(data, columns_list, parameters, fs=1.0):
    """功率谱密度可视化"""
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    for feature in columns_list:
        f, Pxx = signal.welch(data[feature].dropna(), fs=fs)

        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Pxx)
        plt.title(f'Power Spectral Density of {feature}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.grid(True)
        plt.show()


def Hurst_Exponent_Visual(data, columns_list, parameters, max_lag=50):
    """Hurst指数计算与可视化"""
    data = data.copy()
    if len(columns_list)>parameters['cut_length'] and parameters['cut_OR_Not']:
        print("length > ",parameters['cut_length'])
        columns_list = columns_list[:parameters['cut_length']]
    from numpy import cumsum, log, polyfit
    from numpy.random import randn

    def hurst(ts):
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = polyfit(log(lags), log(tau), 1)
        return poly[0] * 2.0

    for feature in columns_list:
        values = data[feature].dropna().values
        H = hurst(values)

        plt.figure(figsize=(10, 6))
        plt.plot(values, label=f'Hurst Exponent = {H:.3f}')
        plt.title(f'{feature} Time Series with Hurst Analysis')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
