---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB卷积动画演示"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-04-08T12:00:00+08:00
lastmod: 2018-04-08T12:00:00+08:00
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
连续函数的卷积过程动画展示，修改函数定义可以做任意函数卷积，注意分段函数的定义要正确。
为方便起见，只计算了[-3,3]区间的函数值
代码如下

```
%任意函数卷积过程演示
clc
clear
%定义函数
f=@(x) (x.*0+1).*(x>=0 & x<1)+0;%0-1的阶跃 高1
g=@(x) (x.*0+2).*(x>=0 & x<1)+0;%0-1的阶跃 高2
%计算并画出f，仅计算[-3,3]
x_of_f=-3:0.01:3;
y_of_f=f(x_of_f);
figure(1)
hold on
grid on
plot(x_of_f,y_of_f,'r');
axis ([-3 3 0 3]) %调整坐标显示范围
xlabel('τ','FontSize',16);
x_of_g=-3:0.01:3;%这个区域下计算g

for t = -3:0.04:3 %定义conv(t)=f(t)*g(t)，下面描绘不同t下的情况，构成动画
    y_of_g=g(t-x_of_g);%tao为自变量反折后平移t，这里是一个向量，一组tao
    y_of_g_plot=plot(x_of_g,y_of_g,'b');%画出g
    pause(0.001);%暂停
    delete(y_of_g_plot);%删除原曲线
    %求卷积，仅在-3，3区间积分即可
    sum=0;
    for tao=-3:0.01:3
        sum=sum+0.01*(f(tao)*g(t-tao));
    end
    disp(sum);
    plot(t,sum,'.');
end
hold off
```
{{<figure src = "0.png" title = "" lightbox = "true">}}
