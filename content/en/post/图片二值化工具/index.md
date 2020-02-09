---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "图片二值化工具"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-01-22T12:00:00+08:00
lastmod: 2018-01-22T12:00:00+08:00
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
####通常直接拍摄书籍纸张总是灰乎乎的，应用opencv自适应阈值二值化可以很方便的将照片转化为清晰的二值化照片，打印出来也不会是黑的，代码如下：

```python
#! python2
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import sys
import os


def handle(imgDir):
    name = imgDir.split(".")[0]
    fmt = imgDir.split(".")[1]
    img = cv2.imread(imgDir, 0)
    # img = cv2.medianBlur(img, 5)  # 中值滤波
    # 自适应阈值二值化
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    #cv2.imshow("img", img)
    cv2.imwrite(name + "_cv." + fmt, img)
    print imgDir + " process successfully!"


for arg in sys.argv:
    print arg
    if arg.split(".")[1] not in ["py", "exe"]:
        try:
            handle(str(arg))
        except:
            print "process " + arg + " failed!"

os.system('pause')

```
####效果
{{<figure src = "0.png" title = "" lightbox = "true">}}
{{<figure src = "1.png" title = "" lightbox = "true">}}


####可直接拖拽想要转换的图来启动，会把转换的图片加_cv后缀保存到原地址处，通过pyinstaller打包成exe就可以跨平台了
