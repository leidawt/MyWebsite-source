---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "运放选型策略"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-09-22T12:00:00+08:00
lastmod: 2018-09-22T12:00:00+08:00
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
### 1.考虑直流指标
{{<figure src = "0.png" title = "" lightbox = "true">}}
##### 1.1输入偏置电流与输入失调电流
定义运算放大器两输入端流进或流出直流电流的平均值为输入偏置电流Ib。由输入级的输入电阻不足造成（input bias current）
输入失调电流是运算放大器两输入端输入偏置电流之差的绝对值(input offest current) 由输入级不对称造成。
这两个参数对输出造成的总误差为
{{<figure src = "1.png" title = "" lightbox = "true">}}
##### 1.2输入失调电压
而为了使Vo＝0 而必须在V+和V－间加入的矫正电压Vos 即被称为运算放大器的输入失调电压（input offset voltage）。
这个参数对输出造成的误差为

Vo=Vos(1+R2/R1)
### 2.考虑交流指标
##### 2.1增益带宽积
闭环增益与带宽的乘积。留100倍余量时增益精度很好。
##### 2.2压摆率
{{<figure src = "2.png" title = "" lightbox = "true">}}
##### 2.3总谐波失真加噪声（THD+N）
相关指标：
信号与噪声加失真比SINAD:    SINAD＝20 log(1/THD+N)  以dB 为单位
有效位数ENOB：   ENOB＝（SINAD－1.76）/ 6.02  （位）


