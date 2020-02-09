---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "机器学习预备-Matplotlib绘图"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-10-15T12:00:00+08:00
lastmod: 2018-10-15T12:00:00+08:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Center"
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
#links:
#  - icon_pack: fab
#    icon: twitter
#    name: Follow
#    url: 'https://twitter.com/Twitter'

---
机器学习预备系列博客记述服务机器学习的使用前导知识
记录下python下绘图的方法
首先引用下cheet sheet
工作流
{{<figure src = "0.png" title = "" lightbox = "true">}}

典型操作
```python
#! python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt  # 基本绘图的引用
from matplotlib import style  # 为使用更漂亮的风格

# 折线图
plt.plot([1, 2, 3], [4, 5, 6], label='first line')
plt.plot([1, 2, 3], [7, 8, 9], label='second line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('demo')
plt.legend()  # 画图例
plt.show()


# 散点图
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [5, 2, 4, 2, 1, 4, 5, 2]
plt.scatter(x, y, label='skitscat', color='k', s=25, marker="o")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

# 从文件绘图
import csv
# 读入文件
x = []
y = []
with open('Matplotlib_basic_data.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))
# 或利用numpy
#x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)
plt.plot(x, y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

# 多图
style.use('fivethirtyeight')  # 使用更漂亮的风格

fig = plt.figure()
ax1 = fig.add_subplot(221)  # 高2 宽2 图号1
ax2 = fig.add_subplot(222)  # 高2 宽2 图号2
x1 = [1, 2, 3]
y1 = [2, 5, 4]
x2 = [1, 2, 3]
y2 = [5, 8, 7]
ax1.plot(x1, y1)
ax2.plot(x2, y2)
plt.show()

```
最后附完整的cheet sheet
{{<figure src = "1.png" title = "" lightbox = "true">}}


