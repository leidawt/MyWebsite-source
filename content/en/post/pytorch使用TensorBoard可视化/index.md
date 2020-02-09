---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "pytorch使用TensorBoard可视化"
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
**借助TensorBoardX,可使用优秀的TensorBoard工具**
doc: https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
首先pip安装TensorBoardX和TensorFlow（cpu版本即可）
### 使用
首先引入并构建writer

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs/exp1')
```
注意程序最后执行writer.close()关闭writer
**之后即可开始对TensorBoardX写入需要的信息了，几种常用功能如下：**
#### 1.画loss曲线
```python
writer.add_scalar('batch_loss', batch_loss, epoch_index) 
```
{{<figure src = "0.png" title = "" lightbox = "true">}}
#### 2.画激活情况
用于检查深层网络里的层激活与权值分布情况，避免梯度消失等
```python
for name, param in model.named_parameters():
    writer.add_histogram(
        name, param.clone().data.numpy(), epoch_index)
```
{{<figure src = "1.png" title = "" lightbox = "true">}}
#### 3.画网络结构图
输入模型和输入尺寸（用于内部函数正确遍历网络）
```python
writer.add_graph(model, t.Tensor(1, 784))
```
{{<figure src = "2.png" title = "" lightbox = "true">}}
#### 4.显示图片

```python
writer.add_image('input', x, 1)
writer.add_image('output', y, 1)
```
{{<figure src = "3.png" title = "" lightbox = "true">}}
### 启动
在py文件所在目录运行：tensorboard --logdir runs 即开启界面
{{<figure src = "4.png" title = "" lightbox = "true">}}

