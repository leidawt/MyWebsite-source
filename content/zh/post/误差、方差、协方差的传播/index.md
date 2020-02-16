---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "误差、方差、协方差的传播"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-02-16T12:00:00+08:00
lastmod: 2020-02-16T12:00:00+08:00
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

# 0. 绝对误差与相对误差
一般的，称$x^\*$为准确值$x$的一个近似值，可定义以下两种常用误差:
## 0.1绝对误差（简称误差）：
绝对误差（简称误差）: $e(x^\*)=x-x^\*$
为方便起见，取其一上界$\epsilon$使满足$|x-x^\*|\le\epsilon$（这个上界不唯一），由此我们最常见的误差写法就可写成$x=x^\*\pm\epsilon$，即准确值$x$必在区间$[x^\*-\epsilon,x^\*+\epsilon]$内。
## 0.2相对误差
考虑真值本身的数量大小，相对误差是衡量精度的更好指标，定义为：
$e_r(x^\*)=\frac{x-x^\*}{x}$，相似的，我们亦可取一个上界为$\epsilon\_r$，称之为相对误差限。
# 1. 误差的传播
误差的传播系指分析在形如$y=f(x\_1,x\_2,...,x\_n)$的关系中，参量误差对变量误差的影响有多大。误差的传播与函数的微分紧密相关，本质是在利用当$\Delta x$不大时，$\Delta y\approx \frac{\partial f}{\partial x}\Delta x$。
若$f$在$(x^\*_1,...x^\*_n)$可微，则不难得到:
{{<figure src = "0.png" title = "" lightbox = "true">}}
{{<figure src = "1.png" title = "" lightbox = "true">}}
这实际上就是一阶泰勒展开。多变量泰勒公式为：
$f(x)=f(\bf{a})+\nabla{f(\bf{a})\cdot(\bf{x}-\bf{a})}+...$
移项，令$\bf a$为$(x^\*_1,...x^\*_n)$并带入$e(x^\*)=x-x^\*$，$e(y^\*)=y-y^\*$即可得误差公式。
按照上面的式子求导，可给出最常见 和 差 积 商 误差公式：
{{<figure src = "2.png" title = "" lightbox = "true">}}
相应的，误差限可如下给出：
{{<figure src = "3.png" title = "" lightbox = "true">}}
# 2. 方差传播
仍考虑形如$y=f(x\_1,x\_2,...,x\_n)$的关系，方差传播既是通过$(x\_1,x\_2,...,x\_n)$的不确定度分析$y$的不确定度。该问题有时也称为方差的合成。
## 2.1 简单线性函数
对于简单的函数关系（如加减等），这个方差传播既可用“随机变量函数的分布”这一手段予以求解。即已知随机变量 X 及它的分布，如何求其函数$Y=g(X)$的分布。这是本科概率统计课程的经典内容，不再赘述，通过累次积分可处理形如$Z=X+Y$这类简单形式的概率密度，之后便可求均值方差等所有统计量。
## 2.2 复杂函数
对绝大多数函数，尤其是非线性函数，一般只能寻求其期望和方差的近似求法。这里解决问题的工具依旧是泰勒展开。
**先考虑简单的一维情形**：设随机变量$X$，其期望和方差分别为$\mu,\sigma^2$，变量$Y=g(X)$是其函数。则有：
{{<figure src = "4.png" title = "" lightbox = "true">}}
证明此关系只需将$Y$在$\mu$处局部线性化即可。
{{<figure src = "5.png" title = "" lightbox = "true">}}
相似的，方差公式的证明可进行一阶展开后得到。虽避免了求积分，但此处导数往往也得不到解析解，可通过数值差分近似导数得到（一般倾向于使用中心差分公式）。

**再看二维情形**。一维到二维是一个质变，从两个变量开始，变量相关性和协方差的概念被引入。设随机变量向量$[X,Y]^T$中的变量$X,Y$的期望，方差分别为$\mu \_x, \mu \_y, \sigma \_x^2, \sigma \_y^2$，并设二元函数$Z=g(X,Y)$。则仍仿照上面单变量的方法，将$Z$在自变量期望处进行局部线性化然后两边同取期望/方差即可，只不过换为了多变量的泰勒展开。
**注： 多变量泰勒展开**
{{<figure src = "6.png" title = "" lightbox = "true">}}
更常见的是向量形式：
{{<figure src = "7.png" title = "" lightbox = "true">}}
最后结果是：
{{<figure src = "8.png" title = "" lightbox = "true">}}
可以看到依赖于变量的协方差。
**除了上面直接将函数局部线性化的方法，亦可以使用类似“自底向下”的方法，从方差原始定义入手得到上面的结果。**
仍设随机变量$X$及其一组观测$\{X\_1,X\_2,...X\_n\}$,$Y$及其一组观测$\{Y\_1,Y\_2,...Y\_n\}$，他们的期望，方差分别为$\mu \_x, \mu \_y, \sigma \_x^2, \sigma \_y^2$，二元函数$Z=g(X,Y)$。
则按方差定义
$$\sigma\_Z=\frac{1}{n-1}\sum\_{i=1}^{n}{(Z\_i-\bar Z)^2}$$

通过对$Z=g(X,Y)$在期望处求一阶泰勒展开，可知$$Z\_i-\bar Z=(X\_i-\bar X)\frac{\partial g}{\partial X}+(Y\_i-\bar Y)\frac{\partial g}{\partial Y}$$带回上式，并带入协方差定义：$$cov(X,Y)=\frac{1}{n-1}\sum\_{i=1}^{n}(X\_i-\bar X)(Y\_i-\bar Y)$$

可化简得：

$$\sigma\_Z^2=\frac{1}{n-1}\sum\_{i=1}^{n}[(X\_i-\bar X)\frac{\partial g}{\partial X}+(Y\_i-\bar Y)\frac{\partial g}{\partial Y}]^2$$

$$=\sigma\_x^2(\frac{\partial g}{\partial X})^2+\sigma\_y^2(\frac{\partial g}{\partial Y})^2+2cov(X,Y)\frac{\partial g}{\partial X}\frac{\partial g}{\partial Y}$$

写成向量形式即为：
$$\sigma\_Z^2=J\Sigma J^T$$

其中$J=[\frac{\partial g}{\partial X},\frac{\partial g}{\partial Y}]$为Jacobian matrix,
中间$\Sigma$是协方差矩阵
{{<figure src = "9.png" title = "" lightbox = "true">}}
一些计算实例：
{{<figure src = "10.png" title = "" lightbox = "true">}}
对于线性函数和二次函数, 由于其二阶以上各阶导数为0, 近似计算公式与严密计算公式等价。对于非线性更强的函数，由于我们是以期望为中心展开的，$X\_i$很多时候并不在展开点$\bar X$“附近”，会有不小的误差，可适时地考虑使用蒙特卡洛模拟暴力计算得到更优的结果。
一个常被用于测试的强非线性函数是$Z=Xe^Y$，图像长这样：
{{<figure src = "11.png" title = "" lightbox = "true">}}
# 3. 协方差传播
当上面问题中的函数值域亦是多维的，方差传播就升格为协方差传播。
**先看线性情形**，设多维随机变量X:
{{<figure src = "12.png" title = "" lightbox = "true">}}
设$Z=[k\_1,k\_2,...,k\_n]X+k\_0$为$X$的一线性函数，根据前面方差传播的知识可知有：
$$E(Z)=K\mu\_x+k\_0 ; D\_{ZZ}=KD\_{XX}K^T$$

将其中的$Z$扩展到多维，即设$Z=[z\_1,z\_2,...,z\_t]^T$，其中每个$z\_i$均是$X$的线性函数（$z\_i=[k_{i,1},...,k_{i,n}]X\_i$）
那么对:
$$Z=KX+K\_0$$

其中$K$为$t\*n$矩阵，$K\_0$为$t\*1$矩阵，依旧有相同的结论：
$$E(Z)=K\mu\_x+k\_0 ; D\_{ZZ}=KD\_{XX}K^T$$

只不过这里$D\_{ZZ}$升格为了$t\*t$矩阵，称之为协方差传播。
若另有Y：
{{<figure src = "13.png" title = "" lightbox = "true">}}
并有关于$Y$的函数$W=FY+F\_0$，则相似的可获知一系列关系常用协方差传播律：
$D\_{ZZ}=KD_{XX}K^T$
$D\_{WW}=FD\_{YY}F^T$
$D\_{ZW}=KD\_{XY}F^T$
$D\_{WZ}=FD\_{YX}K^T$

# 4. 参考文献
https://www.cnas.org.cn/fwzl/images/tc261sc1sysrkfjswyh/tzgg/2015/03/24/70E459F9EA361F6C3F4C675277B5CF3C.pdf
https://wenku.baidu.com/view/960f3cd7b8f67c1cfbd6b826.html
https://wenku.baidu.com/view/4d8b1945581b6bd97f19eaab.html
https://www.ucl.ac.uk/~ucfbpve/geotopes/indexch10.html
概率论与数理统计教程(茆诗松)
