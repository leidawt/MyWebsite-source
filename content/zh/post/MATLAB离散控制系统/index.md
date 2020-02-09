---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB离散控制系统"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2017-12-08T12:00:00+08:00
lastmod: 2017-12-08T12:00:00+08:00
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
## MATLAB离散控制系统仿真常用操作
### 1.离散传递函数构建
#### 通过离散化连续时间传递函数得到
连续时间传递函数s函数用tf构建：
```matlab
Gc=tf([1],[1 1 0]);%参数为分子分母降幂排列S的系数
```
对其离散化：通常在对象前面加上一个零阶保持器{{<figure src = "0.png" title = "" lightbox = "true">}}
```matlab
ans_zoh=c2d(Gc,1,'zoh')%添加零阶保持器离散化T=1
ans_imp=c2d(Gc,1,'imp')%或直接离散化T=1
%后者等效
ilaplace(1/(s^2+s))%拉普拉斯反变换
compose(ans,1*t)%T=1采样，t->T*t
ztrans(ans)%z变换
pretty(vpa(collect(ans)))%整理显示
```
### 2.性能分析
```matlab
rlocus(ans_zoh);%根轨迹
bode(ans_zoh);%伯德图
nyquist(ans_zoh)%奈奎斯特图

```
查看系统根轨迹
可从根轨迹明显看到差异，非理想采样的加入降低了稳定性
{{<figure src = "1.png" title = "" lightbox = "true">}}
奈奎斯特图：
{{<figure src = "2.png" title = "" lightbox = "true">}}
伯德图：
红色：连续
蓝色：经zero保持器采样
黄色：经理想采样
{{<figure src = "3.png" title = "" lightbox = "true">}}
### 3.simulink
先按照连续时间方法搭建系统
然后应用离散化工具，选择离散化方法，采样时间对模型离散化
{{<figure src = "4.png" title = "" lightbox = "true">}}
{{<figure src = "5.png" title = "" lightbox = "true">}}
{{<figure src = "6.png" title = "" lightbox = "true">}}
仿真得到不同采样周期下的情况如下
T=1:
{{<figure src = "7.png" title = "" lightbox = "true">}}
T=0.1；
{{<figure src = "8.png" title = "" lightbox = "true">}}
