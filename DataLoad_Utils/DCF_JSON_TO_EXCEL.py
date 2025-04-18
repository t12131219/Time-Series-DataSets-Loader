import os
import json
import pandas as pd


def merge_json_to_excel(folder_path, output_filename):
    """
    合并文件夹中所有JSON文件中的属性数据到一个Excel表格

    参数：
    folder_path - 包含JSON文件的文件夹路径
    output_filename - 输出的Excel文件名（包括.xlsx扩展名）
    """
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)

            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 获取顶层结构（支持不同结构的JSON文件）
                for top_key in data:
                    if 'attributes' in data[top_key]:
                        # 合并属性数据
                        merged_data.extend(data[top_key]['attributes'])
                        break  # 假设每个文件只有一个包含attributes的顶级键

    # 创建DataFrame并保存为Excel
    df = pd.DataFrame(merged_data)
    df.to_excel(output_filename, index=False)
    return f"成功合并 {len(merged_data)} 条属性数据到 {output_filename}"


# 使用示例
if __name__ == "__main__":
    result = merge_json_to_excel('JSON', '../merged_output.xlsx')
    print(result)