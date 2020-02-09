---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "迁移学习"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2019-01-15T12:00:00+08:00
lastmod: 2019-01-15T12:00:00+08:00
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
# 迁移学习overview
{{<figure src = "0.png" title = "" lightbox = "true">}}
**相关论文与实现：**
https://github.com/jindongwang/transferlearning/tree/master/code
#### 1.源于目标均有标签
微调：最简单的迁移方法。在图像分类里效果显著。常通过微调在ImageNet上训练好的大模型，微调后几层，或更极端的，直接重训最后的全连接层。
例，在猫狗大战数据集里，通过重新训练最后的fc层就可轻松达到99.4%左右的测试集精度。

多任务学习：通过相似任务共享几层隐含层进行。在NLP里效果很好
{{<figure src = "1.png" title = "" lightbox = "true">}}
#### 2.源数据集有标签，目标数据集无标签
有很多方法。
1.域对抗训练方法
基本思想借鉴了GAN的对抗。理论基础是一个好的特征提取应该是对数据域不敏感的，因此设计了由三个子网络构成的NN。其中特征提取器用于提取抽象特征，分类器根据抽象特征给出分类，域分辨器用于分辨抽象特征来自哪个数据域。优化目标是在有高的分类精度的同时，让域分辨器不能分辨抽象特征的来源（意味着特征不但有效且对数据域变更有强鲁棒性）。
https://blog.csdn.net/a1154761720/article/details/51020105
https://github.com/fungtion/DANN
{{<figure src = "2.png" title = "" lightbox = "true">}}
2.zero/one shot learning
是一种极端形式的迁移学习，可迁移到从未处理过的样本。思路和NLP的词向量很相似，通过映射到一定维数的Embedding 空间，然后通过判断在Embedding 空间的距离远近来分类。
#### 3.源无标签，目标数据有标签
这其实是半监督学习。图中提到的self-taught learning具体做法是把稀疏自编码器与Softmax回归分类器串联起来。先用大量无标签数据集进行无监督的自编码器学习，得到特征表达，再用小的有标签数据集借助前面无监督学习到的特征在后边训练一个简单的有监督分类器，比如softmax。这样大大减轻了数据标注工作量。
#### 4.略
