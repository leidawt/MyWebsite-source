---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "python argparse参数解析器使用笔记"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2017-12-13T12:00:00+08:00
lastmod: 2017-12-13T12:00:00+08:00
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
####argparse是一个按照UNIX规范从命令行读取对程序传参的python模块常用使用方式如下
### demo.py
```python
#! python2
# -*- coding:utf-8 -*- 
import argparse # 引入
parser = argparse.ArgumentParser() # 初始化解析器
# 下面列举常用的参数形式
# 添加参数解析 -a 解析为二进制，激活为True, 默认False，添加帮助指导 “a binary arg”
parser.add_argument('-a', action="store_true", default=False, help="a binary arg")
# 添加参数解析 -b 解析为字符串
parser.add_argument('-b', help="a str arg")
# 添加参数解析 -b 解析为int
parser.add_argument('-c', type=int, help="a int arg")
# 添加参数解析 -b 解析为float
parser.add_argument('-d', type=float, help="a float arg")

args = vars(parser.parse_args())# 从命令行读参数，解析到args
print args["a"]
print args["b"]
print args["c"]
print args["d"]
```

### 使用时
```bat
python demo.py -a -b hello -c 123 -d 1.23
```
### 结果

```bat
True
hello
123
1.23
```
### 推荐阅读
<http://blog.xiayf.cn/2013/03/30/argparse/>

<https://docs.python.org/2/howto/argparse.html#introducing-optional-arguments>
