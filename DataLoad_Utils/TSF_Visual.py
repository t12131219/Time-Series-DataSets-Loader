import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import os

from TSF_DataLoader import convert_tsf_to_dataframea_array


# data_dir = 'datasets/monash_datasets/bitcoin_dataset_with_missing_values/bitcoin_dataset_with_missing_values.tsf'
# data_dir = 'datasets/monash_datasets/bitcoin_dataset_without_missing_values/bitcoin_dataset_without_missing_values.tsf'
# data_dir = 'datasets/monash_datasets/covid_deaths_dataset/covid_deaths_dataset.tsf'
# data_dir = 'datasets/monash_datasets/electricity_hourly_dataset/electricity_hourly_dataset.tsf'
data_dir = '../datasets/monash_datasets/australian_electricity_demand_dataset/australian_electricity_demand_dataset.tsf'
# data_dir = 'datasets/monash_datasets/electricity_weekly_dataset/electricity_weekly_dataset.tsf'
# data_dir = 'datasets/monash_datasets/m1_monthly_dataset/m1_monthly_dataset.tsf'
# data_dir = 'datasets/monash_datasets/m1_yearly_dataset/m1_yearly_dataset.tsf'
# data_dir = 'datasets/monash_datasets/m3_yearly_dataset/m3_yearly_dataset.tsf'
# data_dir = 'datasets/monash_datasets/m4_yearly_dataset/m4_yearly_dataset.tsf'
# data_dir = 'datasets/monash_datasets/us_births_dataset/us_births_dataset.tsf'
# data_dir = 'datasets/monash_datasets/saugeenday_dataset/saugeenday_dataset.tsf'
# 使用os.path.basename()从路径中提取文件名（包括扩展名）
file_with_extension = os.path.basename(data_dir)

# 分割字符串并获取不包含扩展名的文件名部分
result_name = file_with_extension.split('.')[0]
data,_, _, _, _  = convert_tsf_to_dataframea_array(data_dir)
print(data.shape)

# 设置保存图像的文件夹路径
# save_folder = "Save_TSF_Demo/"  # 替换为你的文件夹路径
save_folder = "Result/" + result_name + "/" # 替换为你的文件夹路径
# 确保文件夹存在
import os
os.makedirs(save_folder, exist_ok=True)

# 如果 'state' 列不存在，可以注释掉或删除相关代码
# 如果 'series_value' 列是字符串格式，将其转换为列表
# data['series_value'] = data['series_value'].apply(literal_eval)

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


# 保存图像
save_path = os.path.join(save_folder, f'{result_name}.png')
plt.savefig(save_path)

# 添加图例和标签
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Visualization')
plt.grid(True)
plt.show()


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
    save_path = os.path.join(save_folder, f'time_series_{series_name}_{state}.png')
    plt.savefig(save_path)

    # 关闭当前图像以释放内存
    plt.close()

print("所有图像已保存！")