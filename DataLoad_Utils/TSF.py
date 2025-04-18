import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import os

'''
parameters['Save_OR_Not'] = True    # 是否保存结果图
parameters['cut_OR_Not'] = True     # 是否对特征列过长的数据进行截断
parameters['cut_length'] = 30       # 截断长度
parameters['CDF_OR_Not'] = False    # 是否显示CDF图
'''

def TSF_Draw_Visiual_Multi_Feature(data, columns_list, parameters, result_name,result_dir):
    # 绘制可视化图
    plt.figure(figsize=(15, 10))

    for _, row in data.iterrows():
        series_name = row['series_name']
        state = row.get('state', 'N/A')  # 如果 'state' 列不存在，用 'N/A' 替代
        start_timestamp = row['start_timestamp']
        series_value = row['series_value']

        # 假设 series_value 是按时间顺序排列的列表
        # time_points = pd.date_range(start=start_timestamp, periods=len(series_value), freq='Q')  # 按季度划分时间

        # 绘制每条时间序列
        plt.plot( series_value, label=f'{series_name} ({state})', linestyle='-')

    # 确保文件夹存在
    os.makedirs(result_dir, exist_ok=True)
    # 保存图像
    save_path = os.path.join(result_dir, f'{result_name}.png')
    plt.savefig(save_path)

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Visualization')
    plt.grid(True)
    plt.show()

def TSF_Draw_Visiual_Single_Feature(data, columns_list, parameters, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    # 绘制可视化图
    num_series = len(data)  # 获取时间序列的数量
    fig, axs = plt.subplots(num_series, figsize=(15, 5 * num_series))  # 创建子图

    for i, (_, row) in enumerate(data.iterrows()):
        series_name = row['series_name']
        state = row.get('state', 'N/A')  # 如果 'state' 列不存在，用 'N/A' 替代
        start_timestamp = row['start_timestamp']
        series_value = row['series_value']

        # 假设 series_value 是按时间顺序排列的列表
        # time_points = pd.date_range(start=start_timestamp, periods=len(series_value), freq='Q')  # 按季度划分时间

        # 绘制每条时间序列
        axs[i].plot(series_value, label=f'{series_name} ({state})', linestyle='-')
        axs[i].set_title(f'Time Series {series_name} ({state})')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')
        axs[i].grid(True)

    # # 调整布局
    # plt.tight_layout()
    # plt.show()

# -------------------------------------------------
def TSF_Draw_Visiual_Single_Feature_Each_Picture(data, columns_list, parameters, result_dir):

    os.makedirs(result_dir, exist_ok=True)
    # 遍历每一行数据
    for _, row in data.iterrows():
        series_name = row['series_name']
        state = row.get('state', 'None')  # 如果 'state' 列不存在，用 'N/A' 替代
        start_timestamp = row['start_timestamp']
        series_value = row['series_value']

        # 创建一个新的图像
        plt.figure(figsize=(15, 10))

        # 绘制时间序列
        plt.plot(series_value, label=f'{series_name} ({state})', linestyle='-')

        # 添加标题和标签
        plt.title(f'Time Series {series_name} ({state})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)

        # 保存图像
        save_path = os.path.join(result_dir, f'time_series_{series_name}_{state}.png')
        plt.savefig(save_path)

        # 关闭当前图像以释放内存
        plt.close()

    print("所有图像已保存！")