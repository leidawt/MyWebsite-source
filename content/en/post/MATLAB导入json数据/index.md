---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB导入json数据"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2017-12-17T12:00:00+08:00
lastmod: 2017-12-17T12:00:00+08:00
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
#### 首先安装json解码工具箱
在此页面
<https://cn.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files>
下载工具箱zip
解压复制到{安装路径}\MATLAB\R2015b\toolbox下
matlab下添加路径
```matlab
addpath('{安装路径}\MATLAB\R2015b\toolbox\jsonlab-1.5')
```
#### 使用
```matlab
json2data=loadjson('xxx.json')
```
即可导入工作区，解析为struct类型
