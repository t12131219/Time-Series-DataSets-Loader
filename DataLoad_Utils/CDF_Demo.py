



# 假设data是你已经加载的DataFrame，且其中有一个名为'column_name'的列
column_name = 'Zone 1 Power Consumption'  # 替换为你想要绘制CDF的列名

# 计算累积分布函数
cdf_values = np.sort(data[column_name])  # 对数据进行排序
cdf_values2 = np.arange(1, len(cdf_values)+1) / len(cdf_values)

# 绘制CDF图
plt.figure(figsize=(10, 6))
plt.step(cdf_values, cdf_values2, where='post', label=column_name)
plt.title(f'CDF of {column_name}')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()