---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "借助MATLAB与SIMULINK仿真嵌入式C算法"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-03-04T12:00:00+08:00
lastmod: 2018-03-04T12:00:00+08:00
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
## 概述
为在嵌入式开发中碰到的算法验证问题，借助matlab平台可以更方便的调试。如控制算法，可以先验证算法编写的正确性，防止盲目调参的无用功。借助的是SIMULINK 与 S-Function Builder
## 方法
首先保证 matlab MEX部分能正常工作，可以参考 mex -setup相关信息。主要是让mex找到正确的编译器，如：已安装了gcc套件，则只需设置下环境变量：		   setenv('MW_MINGW64_LOC','F:\MinGW64');
即可
S-Function 是一个将c c++ 等编译为simulink模块的工具，吧待检测的代码构建为simulink 标准模块便可借助 simulink 强大功能仿真了。S-Function 有特定的格式，可以手写，这里用更简单的S-Function Builder做。
下面以一个iir滤波器算法为例
1.待验证的iir实现如下，保存为iir_souce.c

```
/**
 * @brief 离散 IIR 滤波器算法，被s function builder
 * 使用并建立iir.c(与builder设置的函数名相同)
 *
 * @param u 输入
 * @param xD 离散变量寄存器， sfuncton builder 提供
 * @return double 输出
 */
double iir(double u, double* xD) {
    double y;
    // IIR 的查分方程： y(n)=0.2x(n)+0.3x(n-1)+0.5x(n-2)
    //定义 xD[0] x(n-1) xD[1] x(n-2)
    y = 0.2 * u + 0.3 * xD[0] + 0.5 * xD[1];
    xD[1] = xD[0];
    xD[0] = y;
    return y;
}
```

2.开启新的simulink，导入S-Function Builder 模块，双击打开编辑
2.1 命名
要求与刚才c文件不同命，不然会覆盖

{{<figure src = "0.png" title = "" lightbox = "true">}}
2.2 离散状态设置
这是s function 特殊性，就是离散化的算法中的x[n-1]这样的历史值需要构建为离散状态。如上图设置两个，给x[n-1] x[n-2]用
2.3 函数输入输出设置
iir只需要单输入单输出 宽度都为1 double(默认的) 格式
{{<figure src = "1.png" title = "" lightbox = "true">}}
2.4 说明要编译的文件
左侧 iir_souce.c 指出要编译的文件 右侧extern double iir(double u, double* xD); 提示一会儿要在后面用到里面的这个函数
{{<figure src = "2.png" title = "" lightbox = "true">}}
2.5 这一步指出实现
y0[0]=iir(u0[0],xD);
u0[0] 是输信号，y0[0]为输出信号，xD是刚才定义的离散状态数组
{{<figure src = "3.png" title = "" lightbox = "true">}}
2.6 编译
如图，默认设置就好，还可以产生TLC用于matlab builder 自动生成代码
{{<figure src = "4.png" title = "" lightbox = "true">}}
2.7测试
构建如下系统：
{{<figure src = "5.png" title = "" lightbox = "true">}}
{{<figure src = "6.png" title = "" lightbox = "true">}}
## 至此，成功将目标代码构建到sinmulink中仿真
#参考
[官方文档](https://cn.mathworks.com/help/simulink/sfg/s-function-builder-dialog-box.html#responsive_offcanvas)
参考书 基于模型的设计及其嵌入式实现
