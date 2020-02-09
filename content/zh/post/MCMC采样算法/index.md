---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MCMC采样算法"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2019-06-20T12:00:00+08:00
lastmod: 2019-06-20T12:00:00+08:00
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
ref:
https://www.cs.ubc.ca/~arnaud/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf
MCMC（Markov chain Monte Carlo）是一类采样方法，起源与1930年代的研究。MCMC模拟是解决某些高维困难问题的唯一有效方法，通过选择统计样本来近似困难组合问题是现代MCMC模拟的核心。
MCMC计技术经常被用于解决高维空间下的积分和优化问题，比如求期望，求目标函数极值等。MCMC方法从目标分布p(x) iid的采样N个样点，那么便可由大数定律保证如下的近似：
{{<figure src = "0.png" title = "" lightbox = "true">}}
但有时候p(x)并不是像高斯分布那样易于采样的形式，此时需要一些更复杂的技术，如拒绝采样，重要性采样和MCMC。
**拒绝采样算法如下所示：**
{{<figure src = "1.png" title = "" lightbox = "true">}}
{{<figure src = "2.png" title = "" lightbox = "true">}}
其中q分布一般取好采样的分布比如均匀分布和高斯分布，称参考分布，显然p q若比较接近那么采样效率会比较好，之后乘系数M使得参考分布能完全包住目标分布，之后的采样操作很巧妙，可以看做是以正比于p的概率接收样点。此方法的一个问题是为满足约束M有时会被迫取的很大，从而导致：
{{<figure src = "3.png" title = "" lightbox = "true">}}
这一问题使得拒绝采样在高维场景下不佳。
https://blog.csdn.net/jteng/article/details/54344766
**重要性采样：**
{{<figure src = "4.png" title = "" lightbox = "true">}}
https://www.jianshu.com/p/3d30070932a8

**MCMC采样：**
马尔科夫链（MC）精髓之一在于定义状态转移的概率只依赖于前一个状态。MC有一个有名的收敛定理指出不管链的初始状态如何，最终状态都将收敛到一个固定的终止分布。（需满足以下条件：可能的状态数是有限的，转移概率固定不变，从任意状态能够转变到任意其他状态）
https://www.cnblogs.com/pinard/p/6632399.html
https://cosx.org/2013/01/lda-math-mcmc-and-gibbs-sampling/
MCMC方法利用了MC链会收敛到确定分布的性质，即如果我们能构造一个转移矩阵为P的马氏链，使得该马氏链的平稳分布恰好是p(x), 那么我们从任何一个初始状态x0出发沿着马氏链转移, 得到一个转移序列 x0,x1,x2,⋯xn,xn+1⋯,， 如果马氏链在第n步已经收敛了，于是我们就得到了p(x)的样本xn,xn+1⋯！！！
上述方法起源于Metropolis，经一些优化后成为了经典的MH-MCMC算法。
关于MH-MCMC的原理详见：
https://zhuanlan.zhihu.com/p/37121528
https://cosx.org/2013/01/lda-math-mcmc-and-gibbs-sampling/
简言之，MH-MCMC目标是设计能收敛到目标分布的MC链的巧妙方法。直接找到理想转移矩阵是几乎不可行的，MH方法引入接受率概念对不完美的转移矩阵进行修正，使得满足收敛条件。
上面资料主要以离散形式进行说明，在连续问题上，MC链的状态转移如下计算，{{<figure src = "5.png" title = "" lightbox = "true">}}
原来的转移矩阵变为了积分核K，经常为高斯分布。这种形式的MCMC在实践中更为多见，积分核K的具体设计是个关键问题，很多MCMC的改进变种都针对积分核K进行(比如著名的吉布斯采样，进来还有用深度学习来做核的)。下图为使用不同方差高斯积分核K的采样结果，展示了因核不佳造成采样效率低和视野窄的问题，这也是设计新核的最主要关注点和最期望改善的问题。
{{<figure src = "6.png" title = "" lightbox = "true">}}
https://zhuanlan.zhihu.com/p/67691581
简介了一种重要的改进算法：HMC，HMC是很多概率建模软件如pymc stan等的默认MCMC方法，工程上使用非常普遍。

