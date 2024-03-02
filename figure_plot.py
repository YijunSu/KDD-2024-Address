import numpy as np
import matplotlib.pyplot as plt

# x轴刻度标签
x_ticks = ["Precision", "Recall", "F1", "Spearman"]
# 柱的宽度
barWidth = 0.25
# 第1个柱的x轴范围（每个柱子的中点）（0, 1, ..., len(x_ticks)-1）
x1 = np.arange(len(x_ticks))
# 第2个柱的x轴范围（每个柱子的中点）
x2 = [x + barWidth for x in x1]
# 第3个柱的x轴范围
x3 = [x + 2*barWidth for x in x1]

# 第1个柱数据
BERT = [0.937, 0.936, 0.936, 0.828]
# 第2个柱数据
GeAttn = [0.98, 0.978, 0.979, 0.861]

# 设置画布大小
plt.figure(figsize=(10, 6))
# 画第1个柱
plt.bar(x1, BERT, color='#ff7f4c', width=barWidth, edgecolor='black', label='SimCSE-BERT_base')
# 画第2个柱
plt.bar(x2, GeAttn, color='#24345A', width=barWidth, hatch='//', edgecolor='black', label='SimCSE-GEAttn_base')

# 给第1个柱数据点加上数值，前两个参数是坐标，第三个是数值，ha和va分别是水平和垂直位置（数据点相对数值）。
for a, b in zip(x1, BERT):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=12)
# 给第2个柱数据点加上数值
for a, b in zip(x2, GeAttn):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=12)

# 画水平横线
# plt.hlines(3, 0, len(x_ticks)-1+barWidth, colors="#000000", linestyles="dashed")

# 添加x轴和y轴刻度标签
plt.xticks([r + barWidth/2 for r in x1], x_ticks, fontsize=18)
plt.yticks(fontsize=18)
# plt.xlim((0, 2))
plt.ylim((0.8, 1.0))

# 添加x轴和y轴标签
# plt.xlabel(u'x_label', fontsize=18)
# plt.ylabel(u'y_label', fontsize=18)

# 标题
# plt.title(u'Title', fontsize=18)

# 图例
plt.legend(fontsize=12, frameon=False)
plt.grid()

# # 保存图片
# plt.savefig('./figure.pdf', bbox_inches='tight')
# 显示图片
plt.show()