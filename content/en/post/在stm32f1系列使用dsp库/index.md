---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "在stm32f1系列使用dsp库"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2017-12-06T12:00:00+08:00
lastmod: 2017-12-06T12:00:00+08:00
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
# 在stm32f1系列使用dsp库
##获取dsp库
在keil mdk 版本中，dsp库集成与runtime environment之中，可以在keil安装目录找到，通常路径：
C:\Keil_v5\ARM\PACK\ARM\CMSIS\4.5.0\CMSIS\DSP_Lib
或者从官网获取：CMSIS-DSP Library for Cortex-M, SC000, and SC300
      Pack: ARM::CMSIS, http://www.keil.com/pack/ARM.CMSIS.4.5.0.pack 
## dsp库内容
{{<figure src = "0.png" title = "" lightbox = "true">}}
dsp库包含常用数学运算，复数，矩阵，三角函数，还有重要的fir滤波器和FFT，非常实用
## 使用dsp库
### 1.1使用runtime environment 包管理器时引入
只需勾选dsp
{{<figure src = "1.png" title = "" lightbox = "true">}}

### 1.2不使用runtime environment 包管理器时引入
此时引入dsp lib 通常因为自己加入了cmX.h内核文件导致错误，因为runtime environment会自动处理依赖，添加内核，这时只需要将内核头文件的文件夹从include path 中移除即可
###2.添加全局宏定义
添加内核定义：在此处添加  ARM_MATH_CM3 宏定义，其他内核按需修改可为CM0 ~ CM4
{{<figure src = "2.png" title = "" lightbox = "true">}}
### 3.头文件
最后引入
```c
#include "arm_math.h"
```
便可以引用了
## 文档与例程

帮助文件位于 
C:\Keil_v5\ARM\PACK\ARM\CMSIS\4.5.0\CMSIS\Documentation\RTX\html\index.html

此文件夹Examples目录
C:\Keil_v5\ARM\PACK\ARM\CMSIS\4.5.0\CMSIS\DSP_Lib
中有大量官方例程可供参考
