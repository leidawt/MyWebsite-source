---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB实时脚本"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2019-09-27T12:00:00+08:00
lastmod: 2019-09-27T12:00:00+08:00
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
自matlab2016a版本以来，matlab多了创建实时脚本的功能。在未来版本中作为mupad的替代。其思想与mathcad相似，企图将文档与程序合二为一。就是在原有m文件上加了交互式图标，富文本功能和控件。格式为mlx。2016a以上版本都可打开，还可以输出为pdf等用于分享。
{{<figure src = "0.png" title = "" lightbox = "true">}}
创建实时脚本很简单，help里有详细记述，这里记录下实时脚本的功能和技巧。
首先是交互式输入公式的能力，如下，有强大的公式录入功能，还支持latex
{{<figure src = "1.png" title = "" lightbox = "true">}}
{{<figure src = "2.png" title = "" lightbox = "true">}}
其次，绘图结果可以进行交互式操作，并生成对应操作的代码。如下：
{{<figure src = "3.png" title = "" lightbox = "true">}}
还可以插入箭头，加网格，图例等，比用代码写要更方便快捷，只需之后操作生成的代码更新到code即可
{{<figure src = "4.png" title = "" lightbox = "true">}}
此外**一个十分重要特性**是对公式结果的展示，比以前的pretty更加优秀的是，其可进行手写化展示。以后利用matlab进行符号推导更加舒服了。
{{<figure src = "5.png" title = "" lightbox = "true">}}
自2018a开始，又增加了控件功能，可嵌入代码之中，通过拖动和下拉菜单选项进行取值更改，之后会自动刷新和执行一遍程序。
{{<figure src = "6.png" title = "" lightbox = "true">}}
