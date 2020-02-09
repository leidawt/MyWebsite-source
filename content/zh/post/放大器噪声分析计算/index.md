---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "放大器噪声分析计算"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-06-06T12:00:00+08:00
lastmod: 2018-06-06T12:00:00+08:00
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
1.何为运放噪声
放大器的噪声模型如下
{{<figure src = "0.png" title = "" lightbox = "true">}}
大致有三部分组成，运放电压噪声，运放电流噪声，和反馈电阻产生的噪声。三者的平方和开根号就是总噪声。在处理精密信号时，显然噪声的问题就很关键了。
2.运放噪声基础
{{<figure src = "1.png" title = "" lightbox = "true">}}
运放的噪声信息以上图这样的形式给出，这是一个频率谱密度图。因为噪声和频率相关，此图表征了噪声包含的各个频率分量的大小。图分两个区间，前面线性下降部分称为闪烁噪声，后半部分为宽白噪声。对此频谱积分得到总噪声。再讨论噪声带宽问题，显然上图直接积分到正无穷是无限大，显然不合理。其实噪声也是有带宽的。这个带宽决定于运放的带宽。和运放带宽-3db定义不同，噪声定义在完全截止处。
{{<figure src = "2.png" title = "" lightbox = "true">}}
因此存在经验修正系数，如下。
{{<figure src = "3.png" title = "" lightbox = "true">}}
注意普通放大器看做一个一阶滤波器。
2.实战噪声计算
以opa842构成的10倍放大器为例子
首先从手册找出增益带宽积算出截止频率（这个频率会做为上面的频谱积分的上限）
{{<figure src = "4.png" title = "" lightbox = "true">}}
所以带宽为200/10=20MHz
乘上1.57修正为31.4MHz
接下来找出噪声参数
{{<figure src = "5.png" title = "" lightbox = "true">}}
再找出噪声频谱图
{{<figure src = "6.png" title = "" lightbox = "true">}}

下面的工作交给Bruce Trump的一款计算软件了（网上可下载），就是个excel表
先算闪烁噪声
{{<figure src = "7.png" title = "" lightbox = "true">}}
左上角黄色格子依次填入闪烁噪声：100Hz处20nv/根号赫兹，以及白噪声2.6nv/根号赫兹。都可以从下图读出
{{<figure src = "8.png" title = "" lightbox = "true">}}
右下角写闪烁噪声和白噪声的转折频率。前面的默认写0.1Hz，后面写刚才算出的频谱31.4MHz。
得到如下的结果
{{<figure src = "9.png" title = "" lightbox = "true">}}
接下来切换到总噪声计算页面。由说明书填写数据。
{{<figure src = "10.png" title = "" lightbox = "true">}}
{{<figure src = "11.png" title = "" lightbox = "true">}}
得到结果了：
{{<figure src = "12.png" title = "" lightbox = "true">}}
这个数再乘根号下31.4MHz就是总电压噪声169微伏，对于精密电路相当的大了。还可以看到绝大部分噪声来自电压噪声。
