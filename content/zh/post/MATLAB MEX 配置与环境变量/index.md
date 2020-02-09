---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB MEX 配置与环境变量"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-03-03T12:00:00+08:00
lastmod: 2018-03-03T12:00:00+08:00
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

通过设置MATLAB内部环境变量可以让MEX找到已经安装的编译器
#### MATLAB环境变量

```
%示例
%设置并检索环境变量 TEMP 的新值：

setenv('TEMP', 'C:\TEMP');
getenv('TEMP')
#将 Perl\bin 文件夹附加到您的系统 PATH 变量：

setenv('PATH', [getenv('PATH') ';D:\Perl\bin']);
```
####MEX编译器的环境变量设置
```
%可以这样初始化,选择编译器
mex -setup -v;
%若环境变量不正确则更改即可，如：
setenv('MW_MINGW64_LOC','F:\MinGW64');
```
