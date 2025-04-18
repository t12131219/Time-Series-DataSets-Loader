from TS_DataLoader import load_from_tsfile_to_dataframe
import os
import pandas as pd

data_dir = 'datasets/Monash_UEA_UCR_Regression_Archive/AppliancesEnergy/AppliancesEnergy_TEST.ts'
# 使用os.path.basename()从路径中提取文件名（包括扩展名）
file_with_extension = os.path.basename(data_dir)

# 分割字符串并获取不包含扩展名的文件名部分
result_name = file_with_extension.split('.')[0]
data= load_from_tsfile_to_dataframe(data_dir)

# df = pd.DataFrame(data)

# print(data.head())


print(data[0])
