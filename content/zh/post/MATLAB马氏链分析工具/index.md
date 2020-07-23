---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB马氏链分析工具"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-07-22T12:00:00+08:00
lastmod: 2020-07-22T12:00:00+08:00
featured: false
draft: false
markup: blackfriday

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
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
{{% toc %}}
本文整理下齐次有限状态离散时间马氏链的相关基础内容并及MATLAB中提供的与之相关的性质。
# 基本性质
为进行状态分类，先引入一组重要性质和定义
{{<figure src = "0.png" title = "" lightbox = "true">}}
{{<figure src = "1.png" title = "" lightbox = "true">}}{{<figure src = "2.png" title = "" lightbox = "true">}}
{{<figure src = "3.png" title = "" lightbox = "true">}}
{{<figure src = "4.png" title = "" lightbox = "true">}}
{{<figure src = "5.png" title = "" lightbox = "true">}}
{{<figure src = "6.png" title = "" lightbox = "true">}}

{{<figure src = "7.png" title = "" lightbox = "true">}}
# 平稳分布
{{<figure src = "8.png" title = "" lightbox = "true">}}
式子7-94为平衡方程：$\pi=\pi\*P$
不可约且正常返的马氏链一定存在平稳分布，更一般的，只要马氏链存在一个闭的不可约子集，并且该集合中的状态均是正常返的，则存在平稳分布。
{{<figure src = "9.png" title = "" lightbox = "true">}}
# 混合时间（mixing time）
在概率论中，马尔可夫链的混合时间是马尔可夫链“接近”其稳态分布的所需时间。
{{<figure src = "10.png" title = "" lightbox = "true">}}
对于遍历链，任何初始分布都以第二大特征值模量（SLEM）$\mu$确定的速率收敛到平稳分布。 谱间隙$1-\mu$，提供了一种可视化的测量方法，具有较大的间隙（较小的SLEM圆），可产生更快的收敛。matlab中估计mixing time的式子是：
$$tMix=-\frac{1}{\log{\mu}}$$

# MATLAB马氏链工具包
MATLAB在Econometrics Toolbox中提供[dtmc](https://www.mathworks.com/help/econ/markov-chain-models.html)类，可绘制状态转移图、判断遍历性等等
1. 生成马氏链
直接输入一步状态转移矩阵即可
	```
	P = [ 0   0  1/2 1/4 1/4  0   0 ;
	      0   0  1/3  0  2/3  0   0 ;
	      0   0   0   0   0  1/3 2/3;
	      0   0   0   0   0  1/2 1/2;
	      0   0   0   0   0  3/4 1/4;
	     1/2 1/2  0   0   0   0   0 ;
	     1/4 3/4  0   0   0   0   0 ];
	mc = dtmc(P);
	```
2. 状态类的判断
	通过绘制状态转移图即可判断马氏链中state类型：
	```
	graphplot(mc,'ColorNodes',true);
	```
{{<figure src = "11.png" title = "" lightbox = "true">}}

3. 可约性、遍历性
	```
	tfRed = isreducible(mc)
	tfErg = isergodic(mc)
	```
4. 周期性
	周期性可以状态分布概率直观的看出来
	```
	X1 = redistribute(mc,20);
	figure;
	distplot(mc,X1);
	```
{{<figure src = "12.png" title = "" lightbox = "true">}}

5. 平稳分布及混合时间
	直接调用下面命令即可
	```
	[xFix,tMix] = asymptotics(mc)
	```
	为观察收敛速度，可绘制特征图
	```
	figure;
	eigplot(mc);
	```
{{<figure src = "13.png" title = "" lightbox = "true">}}
其中粉色阴影表示最大特征值和第二大特征值之间的gap，越大则收敛越快（mixing time越小），若存在周期=k，则半径为1的圆上会有k个点，即存在k个模为1的特征值。
# 参考
https://www.mathworks.com/help/econ/markov-process-models.html?s_tid=CRUX_lftnav
《随机过程及其应用》第二版 陆大金
