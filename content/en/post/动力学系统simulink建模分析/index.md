---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "动力学系统simulink建模分析"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-10-14T12:00:00+08:00
lastmod: 2018-10-14T12:00:00+08:00
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
主要简介动力学系统的建模理论与simulink工程实现，并给出仿真slx文件下载
### simulink实用技巧
为更好的建立模型，归纳了以下技巧
1.双击信号线可对其命名，信号命名后便于监视绘图
{{<figure src = "0.png" title = "" lightbox = "true">}}
2.用信号监视器比scope要更加方便
{{<figure src = "1.png" title = "" lightbox = "true">}}
如上添加监视后可在此查看
{{<figure src = "2.png" title = "" lightbox = "true">}}
3.关于求解器，通常使用自动选择的求解器。常见ode45，这是基于泰勒展开的方法。会自动根据4 5阶的结果差选择仿真步长。在变化快处计算密集，缓慢处计算稀疏。一般限制其最大允许仿真步长保证时间分辨率
{{<figure src = "3.png" title = "" lightbox = "true">}}
4.为了模拟真实世界传感器从连续世界中读取的离散值，一般用零阶保持器离散结果。
{{<figure src = "4.png" title = "" lightbox = "true">}}
5.simulink模型层次
一个好的模型要有清晰的层次结构，使用子系统模块先搭建好顶层模型，再对各个模块细节进行实现
{{<figure src = "5.png" title = "" lightbox = "true">}}
官方样例清晰的表达了一个扫地机器人建模的最佳实践
https://www.mathworks.com/help/simulink/gs/architecture-and-interfaces.html
{{<figure src = "6.png" title = "" lightbox = "true">}}
6.simscape
为更加便利化物理系统，可使用simscape
下面的例子对其构成进行了基本简介
{{<figure src = "7.png" title = "" lightbox = "true">}}
### 基本理论
机械系统常常由下面三个元件构成
1.惯性元件
2.弹性元件
	F=k*(derta x)
3.阻尼表示如下
{{<figure src = "8.png" title = "" lightbox = "true">}}
要特别注意偶数次的符号，一般如下处理
{{<figure src = "9.png" title = "" lightbox = "true">}}
下面是重要的基本物理原理
{{<figure src = "10.png" title = "" lightbox = "true">}}
{{<figure src = "11.png" title = "" lightbox = "true">}}
系统分析的常用数学知识有
虚位移原理
http://netedu.xauat.edu.cn/jpkc/netclass/jpkc/lx/jxzy/wljc/15.pdf
泛函与变分原理
http://www.cad.zju.edu.cn/home/zhx/FAVM/1.pdf
自动控制原理
现代控制原理


### 弹球例子
仿真一个从10m处以15m/s向上抛出的弹性球，每次反弹速度衰减为0.8倍
首先由力学建立方程
{{<figure src = "12.png" title = "" lightbox = "true">}}
之后搭建仿真如下，注意积分初值
{{<figure src = "13.png" title = "" lightbox = "true">}}
仿真后从查看器查看速度和位移图如下
{{<figure src = "14.png" title = "" lightbox = "true">}}
### 状态空间例子
{{<figure src = "15.png" title = "" lightbox = "true">}}
非线性例子仿真波形
{{<figure src = "16.png" title = "" lightbox = "true">}}
线性仿真波形
{{<figure src = "17.png" title = "" lightbox = "true">}}

### 一个磁悬浮的例子
现仿真一个电磁铁吸引小球的系统。
{{<figure src = "18.png" title = "" lightbox = "true">}}
如图，对求做受力分析建立模型，其中忽略空气阻力，将电磁力简化成仅与输入电流和电磁铁与球距离线性相关。
这里的建模要注意避免代数环问题，原子系统在仿真中会被认为是直通而在闭环中造成代数环问题，所有这里的子系统不可使用原子子系统，而应使用子系统
{{<figure src = "19.png" title = "" lightbox = "true">}}
用自动整定进行pid参数选择
首先先手动粗略调节使得系统为稳定系统
之后打开整定器
{{<figure src = "20.png" title = "" lightbox = "true">}}
系统特殊，自动线性化失败。通过新插入plant选择稳定区让系统去识别即可
{{<figure src = "21.png" title = "" lightbox = "true">}}
选择稍靠后的稳定区后即可成功线性化
{{<figure src = "22.png" title = "" lightbox = "true">}}
即得最终仿真波形
{{<figure src = "23.png" title = "" lightbox = "true">}}
### 倒立摆
1.考虑转动
利用是刚体的假设，有
M（力矩）=J（转动惯量）*a（角加速度）
2.考虑质心受力
{{<figure src = "24.png" title = "" lightbox = "true">}}
{{<figure src = "25.png" title = "" lightbox = "true">}}
不难列写如下的微分方程，其中XY指力,m杆质量 M车质量：
对杆
{{<figure src = "26.png" title = "" lightbox = "true">}}
{{<figure src = "27.png" title = "" lightbox = "true">}}
{{<figure src = "28.png" title = "" lightbox = "true">}}
对小车
{{<figure src = "29.png" title = "" lightbox = "true">}}
简化：令sin(x)=x cos(x)=1
消元导出
{{<figure src = "30.png" title = "" lightbox = "true">}}
令m=0.1 M=1 l=1 j=0.003 g=10
{{<figure src = "31.png" title = "" lightbox = "true">}}
取x theta 为输出
{{<figure src = "32.png" title = "" lightbox = "true">}}
模型得到之后，进行控制，即极点配置，关键是得到反馈阵K

matlab计算如下
A=[0 1 0 0;0 0 -0.8834 0;0 0 0 1;0 0 19.4346 0];
B=[0;0.9893;0;-1.7667];
P=[-2+2*sqrt(3)*1i,-2-2*sqrt(3)*1i,-10,-10];
K=acker(A,B,P)

之后再simulink中搭建如下
{{<figure src = "33.png" title = "" lightbox = "true">}}
其中gain模块如下配置成矩阵形式
{{<figure src = "34.png" title = "" lightbox = "true">}}
最终的控制效果极佳，角度变化很小
{{<figure src = "35.png" title = "" lightbox = "true">}}
这个模型是实际可用的，不需再加观测器，两只编码器就可测得所有参数
### 下载
pan
链接: https://pan.baidu.com/s/1y0DYSMCjhJWX7kCLcn_mKA 提取码: vwgy

