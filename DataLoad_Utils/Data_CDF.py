import pandas as pd
import json
import os
from datetime import datetime
import numpy as np


# def analyze_dataset(df, dataset_name):
#     df = df.copy()
#     analysis = {
#         "dataset_name": dataset_name,
#         "attributes": [],
#         "time_info": {
#             "time_step": None,
#             "time_span": None
#         }
#     }
#
#     # 处理列名（兼容无列名的情况）
#     if df.columns.dtype == 'int64':
#         df.columns = [f'col_{i}' for i in df.columns]
#
#     # 分析各列属性
#     for col in df.columns:
#         col_info = {"name": col}
#
#         # 类型判断（增强型类型检测）
#         dtype = str(df[col].dtype)
#         if dtype.startswith('datetime'):
#             col_info["type"] = "date"
#         elif np.issubdtype(df[col].dtype, np.integer):
#             col_info["type"] = "int"
#         elif np.issubdtype(df[col].dtype, np.floating):
#             col_info["type"] = "float"
#         else:
#             # 增强日期格式检测逻辑
#             is_string_column = df[col].apply(lambda x: isinstance(x, str)).all()
#             col_info["type"] = "string" if is_string_column else "mixed"
#
#             # 如果是纯字符串列，尝试解析为日期
#             if is_string_column:
#                 dates = pd.to_datetime(df[col], errors='coerce')
#                 valid_ratio = dates.notna().mean()
#
#                 # 当有效日期比例超过90%时视为日期列
#                 if valid_ratio > 0.9:
#                     col_info["type"] = "date"
#                     min_date = dates.min()
#                     max_date = dates.max()
#                     col_info["min"] = min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else None
#                     col_info["max"] = max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else None
#
#         # 统计信息
#         if col_info["type"] == "date":
#             if "min" not in col_info:  # 处理原生日期类型
#                 dates = pd.to_datetime(df[col])
#                 min_date = dates.min()
#                 max_date = dates.max()
#                 col_info["min"] = min_date.strftime('%Y-%m-%d')
#                 col_info["max"] = max_date.strftime('%Y-%m-%d')
#         elif col_info["type"] in ["int", "float"]:
#             col_info["min"] = float(df[col].min())
#             col_info["max"] = float(df[col].max())
#         elif col_info["type"] == "string" and "min" not in col_info:  # 非日期字符串
#             col_info["options"] = sorted(df[col].dropna().astype(str).unique().tolist())
#
#         analysis["attributes"].append(col_info)
#
#     # 时间序列分析（基于类型而非列名）
#     date_columns = [col["name"] for col in analysis["attributes"] if col["type"] == "date"]
#     if date_columns:
#         date_col = date_columns[0]
#         try:
#             dates = pd.to_datetime(df[date_col])
#             time_diffs = dates.diff().dropna()
#
#             if not time_diffs.empty:
#                 # 计算时间步长（取众数）
#                 mode_step = time_diffs.mode()[0]
#                 analysis["time_info"]["time_step"] = {
#                     "value": mode_step.total_seconds(),
#                     "unit": "seconds"
#                 }
#
#                 # 计算时间跨度
#                 time_span = dates.max() - dates.min()
#                 analysis["time_info"]["time_span"] = {
#                     "value": time_span.total_seconds(),
#                     "unit": "seconds"
#                 }
#         except Exception as e:
#             print(f"日期解析失败: {str(e)}")
#
#     return analysis
def analyze_dataset(df, dataset_name):
    df = df.copy()
    analysis = {
        "dataset_name": dataset_name,
        "attributes": [],
        "time_info": {
            "time_step": None,
            "time_span": None,
            "num_samples": None
        }
    }

    # 处理列名（兼容无列名的情况）
    if df.columns.dtype == 'int64':
        df.columns = [f'col_{i}' for i in df.columns]

    # 分析各列属性
    for col in df.columns:
        col_info = {"name": col}

        # 类型判断（增强型类型检测）
        dtype = str(df[col].dtype)
        if dtype.startswith('datetime'):
            col_info["type"] = "date"
        elif np.issubdtype(df[col].dtype, np.integer):
            col_info["type"] = "int"
        elif np.issubdtype(df[col].dtype, np.floating):
            col_info["type"] = "float"
        else:
            # 增强日期格式检测逻辑
            is_string_column = df[col].apply(lambda x: isinstance(x, str)).all()
            col_info["type"] = "string" if is_string_column else "mixed"

            # 如果是纯字符串列，尝试解析为日期
            if is_string_column:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_ratio = dates.notna().mean()

                # 当有效日期比例超过90%时视为日期列
                if valid_ratio > 0.9:
                    col_info["type"] = "date"
                    min_date = dates.min()
                    max_date = dates.max()
                    col_info["min"] = min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else None
                    col_info["max"] = max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else None

        # 统计信息
        if col_info["type"] == "date":
            if "min" not in col_info:  # 处理原生日期类型
                dates = pd.to_datetime(df[col])
                min_date = dates.min()
                max_date = dates.max()
                col_info["min"] = min_date.strftime('%Y-%m-%d')
                col_info["max"] = max_date.strftime('%Y-%m-%d')
        elif col_info["type"] in ["int", "float"]:
            col_info["min"] = float(df[col].min())
            col_info["max"] = float(df[col].max())
            col_info["mean"] = float(df[col].mean())
            col_info["variance"] = float(df[col].var())
        elif col_info["type"] == "string" and "min" not in col_info:  # 非日期字符串
            col_info["options"] = sorted(df[col].dropna().astype(str).unique().tolist())

        analysis["attributes"].append(col_info)

    # 时间序列分析（基于类型而非列名）
    date_columns = [col["name"] for col in analysis["attributes"] if col["type"] == "date"]
    if date_columns:
        date_col = date_columns[0]
        try:
            dates = pd.to_datetime(df[date_col])
            time_diffs = dates.diff().dropna()

            if not time_diffs.empty:
                # 计算时间步长（取众数）
                mode_step = time_diffs.mode()[0]
                analysis["time_info"]["time_step"] = {
                    "value": mode_step.total_seconds(),
                    "unit": "seconds"
                }

                # 计算时间跨度
                time_span = dates.max() - dates.min()
                analysis["time_info"]["time_span"] = {
                    "value": time_span.total_seconds(),
                    "unit": "seconds"
                }

                # 数据点数量
                analysis["time_info"]["num_samples"] = len(dates)
        except Exception as e:
            print(f"日期解析失败: {str(e)}")

    return analysis

# def analysis_to_excel(analysis, output_file):
#     # 确保输出目录存在
#     output_dir = os.path.dirname(output_file)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created directory: {output_dir}")
#
#     # 创建一个 Excel 写入器
#     with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
#         # 数据集信息
#         dataset_info = {
#             "Dataset Name": [analysis["dataset_name"]],
#             "Time Step (seconds)": [analysis["time_info"]["time_step"]["value"] if analysis["time_info"]["time_step"] else None],
#             "Time Span (seconds)": [analysis["time_info"]["time_span"]["value"] if analysis["time_info"]["time_span"] else None]
#         }
#         df_dataset_info = pd.DataFrame(dataset_info)
#         df_dataset_info.to_excel(writer, sheet_name='Dataset Info', index=False)
#
#         # 列属性
#         attributes = analysis["attributes"]
#         df_attributes = pd.DataFrame(attributes)
#
#         # 动态选择列
#         columns_to_include = ["name", "type"]
#         for col in ["min", "max", "options"]:
#             if any(col in attr for attr in attributes):
#                 columns_to_include.append(col)
#
#         # 重新排序列
#         df_attributes = df_attributes[columns_to_include]
#
#         # 填充空值
#         df_attributes.fillna("N/A", inplace=True)
#
#         # 保存到 Excel
#         df_attributes.to_excel(writer, sheet_name='Attributes', index=False)
#
#     print(f"Excel file saved to {output_file}")
def analysis_to_excel(analysis, output_file):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 创建一个 Excel 写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 数据集信息
        dataset_info = {
            "Dataset Name": [analysis["dataset_name"]],
            "Time Step (seconds)": [analysis["time_info"]["time_step"]["value"] if analysis["time_info"]["time_step"] else None],
            "Time Span (seconds)": [analysis["time_info"]["time_span"]["value"] if analysis["time_info"]["time_span"] else None],
            "Number of Samples": [analysis["time_info"]["num_samples"] if analysis["time_info"]["num_samples"] else None]
        }
        df_dataset_info = pd.DataFrame(dataset_info)
        df_dataset_info.to_excel(writer, sheet_name='Dataset Info', index=False)

        # 列属性
        attributes = analysis["attributes"]
        df_attributes = pd.DataFrame(attributes)

        # 动态选择列
        columns_to_include = ["name", "type"]
        for col in ["min", "max", "mean", "variance", "options"]:
            if any(col in attr for attr in attributes):
                columns_to_include.append(col)

        # 重新排序列
        df_attributes = df_attributes[columns_to_include]

        # 填充空值
        df_attributes.fillna("N/A", inplace=True)

        # 保存到 Excel
        df_attributes.to_excel(writer, sheet_name='Attributes', index=False)

    print(f"Excel file saved to {output_file}")

def analyze_dataset_tsf(input_rows, dataset_name):
    def process_data(input_rows):
        """
        处理原始数据，生成符合save_analysis_to_json函数要求的结构
        :param input_rows: 输入数据列表（字典结构，包含series_name/state/start_timestamp/series_value字段）
        :return: 格式化后的results列表
        """
        results = []
        for row in input_rows:
            # 处理数值列（兼容字符串形式的列表）
            series_value = row["series_value"]
            if isinstance(series_value, str):
                # 清理省略号并转换为列表
                series_value = json.loads(series_value.replace("...", "").strip())

            # 计算极值
            max_val = max(series_value)
            min_val = min(series_value)

            # 构建结果结构
            result = {
                "dataset_name": row["series_name"],
                "attributes": {
                    row["state"]: {
                        "max": float(max_val),
                        "min": float(min_val)
                    }
                },
                "time_info": row["start_timestamp"]
            }
            results.append(result)
        return results

def save_analysis_to_json(results, output_file):
    formatted = {}
    for result in results:
        dataset_name = result["dataset_name"]
        formatted[dataset_name] = {
            "attributes": result["attributes"],
            "time_info": result["time_info"]
        }

    # 分离目录路径和文件名
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录
        print(f"目录 {output_dir} 已创建。")

    # 写入文件
    with open(output_file, 'w') as f:
        json.dump(formatted, f, indent=2, default=str)


# # 使用示例
# if __name__ == "__main__":
#     # 示例数据集1（带日期列）
#     dates = pd.date_range('2023-01-01', periods=5, freq='D')
#     df1 = pd.DataFrame({
#         'Date': dates,
#         'Temperature': [22.3, 23.5, 24.0, 25.1, 26.0],
#         'Status': ['OK', 'OK', 'Warning', 'OK', 'Critical']
#     })
#
#     # 示例数据集2（数值型数据）
#     df2 = pd.DataFrame({
#         'col_0': [1, 2, 3],
#         'col_1': [1.1, 2.2, 3.3]
#     })
#
#     # 执行分析
#     analysis1 = analyze_dataset(df1, "temperature_data")
#     analysis2 = analyze_dataset(df2, "numeric_data")
#
#     # 保存结果
#     save_analysis_to_json([analysis1, analysis2], "dataset_analysis.json")
