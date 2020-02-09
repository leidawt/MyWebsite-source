---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "机器学习预备-numpy"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-10-16T12:00:00+08:00
lastmod: 2018-10-16T12:00:00+08:00
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
@[toc]
### 文档 
https://docs.scipy.org/doc/numpy/reference/
### 先导
**引入**
import numpy as np
**numpy 数据结构-ndarray**
numpy 使用的数组类是 ndarray
{{<figure src = "0.png" title = "" lightbox = "true">}}
一些重要属性如下：
ndarray.ndim 维数
ndarray.shape 返回(n, m)，n行 m列
ndarray.dtype 类型
**numpy 数据结构-mat**
mat是ndarray的派生，进行矩阵运算比ndarray方便
a=np.mat('4 3; 2 1')
a=np.mat(np.array([[1,2],[3,4]]))
mat 的*重载为矩阵乘法
求逆a.I
### numpy常量
numpy.inf
numpy.nan
numpy.e
numpy.pi
### 创建数组（矩阵）

```python
a = np.array([2,3,4])
a = np.array([[1, 1], [1, 1]])
#指定类型
np.array( [ [1,2], [3,4] ], dtype=complex )
np.zeros( (3,4) )
np.ones( (2,3,4), dtype=np.int16 )
np.empty( (2,3) )  #值不初始化，为内存乱值
#创建数字序列
np.arange( 10, 30, 5 ) #array([10, 15, 20, 25])
np.arange(15).reshape(3, 5)
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14]])
```
附：
全部类型
{{<figure src = "1.png" title = "" lightbox = "true">}}
### 操作数组（矩阵）
常见操作符均已重载，其中注意：*分配成了逐一乘（matlab中.\*），矩阵乘法采用**np.dot(A, B)**
**拷贝**：d = a.copy()
在不同数组类型之间的操作，结果数组的类型趋于更普通或者更精确的一种
array的索引，切片和迭代与python[]同
**切片**
{{<figure src = "2.png" title = "" lightbox = "true">}}
**上下拼接**
np.vstack((a,b))
**左右拼接**
np.hstack((a,b))
### 通用数学函数
https://docs.scipy.org/doc/numpy/reference/ufuncs.html
### 广播规则*
当向量和矩阵结构不匹配响应运算时，会启用广播规则处理,可认为是一种自动补全机制
{{<figure src = "3.png" title = "" lightbox = "true">}}
### 索引
{{<figure src = "4.png" title = "" lightbox = "true">}}
### axis
axis=0，对每列操作
axis=1，对每行操作
如对
{{<figure src = "5.png" title = "" lightbox = "true">}}
{{<figure src = "6.png" title = "" lightbox = "true">}}
### 排序
若x为(n,)
sorted_indices = np.argsort(x)#根据x产生升序排序索引
sorted_indices = np.argsort(-x)#根据x产生降序排序索引
这样使用sorted_indices 来排序其他array
若y为(n,m)
y=y[sorted_indices,:]



