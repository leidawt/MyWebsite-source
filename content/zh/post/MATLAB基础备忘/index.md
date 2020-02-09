---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MATLAB基础备忘"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-02-11T12:00:00+08:00
lastmod: 2018-02-11T12:00:00+08:00
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
# 1.特殊运算符
.*	  数组乘法 （逐位相乘 .的作用，下同）
.^   数组求幂运算符
 \	  矩阵左除
 /	  矩阵右除
 .\	  阵列左除
 ./	  阵列右除
 %	注释标志
# 2.矩阵、向量的创建
r = [7 8 9 10 11] %行向量
r = [7 8 9 10 11]' %列向量
m = [1 2 3; 4 5 6; 7 8 9] %矩阵
# 3. 常用命令
## 系统
clc	清除命令窗口
clear	从内存中删除变量
help	搜索帮助主题
disp	显示一个数组或字符串的内容
input	显示提示并等待输入
MATLAB向量，矩阵和阵列命令

## 用于数组、矩阵、向量
cat	连接数组
find	查找非零元素的索引
**length	计算元素数量**
linspace	创建间隔向量
logspace	创建对数间隔向量
**max	返回最大元素**
**min	返回最小元素**
prod	计算数组元素的连乘积
reshape	重新调整矩阵的行数、列数、维数
**size	计算数组大小**
**sort	排序每个列**
**sum	每列相加**
eye	创建一个单位矩阵
ones	生成全1矩阵
zeros	生成零矩阵
cross	计算矩阵交叉乘积
dot	计算矩阵点积
**det	计算数组的行列式**
**inv	计算矩阵的逆**
pinv	计算矩阵的伪逆
rank	计算矩阵的秩
rref	将矩阵化成行最简形
cell	创建单元数组
celldisp	显示单元数组
cellplot	显示单元数组的图形表示
num2cell	将数值阵列转化为异质阵列
deal	匹配输入和输出列表
iscell	判断是否为元胞类型 
# 4. 逻辑语句
## if
```matlab
a=1;
if a<2
	disp('a<2');
end

a=1;
if a<2
    disp('a<2');
elseif a<3
    disp('a<3');
else
    disp('a>=2');
end
```
## while
```matlab
a=0;
while a<20
    disp(a);
    a=a+1;
end
```
## for
```matlab
for a = 10:20 
   disp(a);
end
for a = 1.0: -0.1: 0.0
   disp(a);
end
for a = [24,18,17,23,28]
   disp(a);
end
```
# 5.向量矩阵的索引与操作
##向量
```matlab
r=[1 2 3 4];
a=r(1);%a=1
b=r(1:2);%a=[1 2]
c=r(:);%c=[1;2;3;4]
%追加
a=[1 2];
b=[3 4];
c=[a,b];%c=[1 2 3 4]
c=[a;b];%c=[1 2;3 4]
dot(a, b);%点乘
```
##矩阵
```matlab
m=[1 2 3;4 5 6;7 8 9];
a=m(1,2);%2
a=m(:,1:2);%前两列
m(:,1)=[];%删除第一列
mm=[m,m];%矩阵组合
randm=rand(3, 5);%0-1均匀分布的随机3*5矩阵
```
#6.函数
```matlab
%add_two_num.m
function y=add_two_num(a,b)
y=a+b;
%隐函数
power = @(x, n) x.^n;
result1 = power(7, 3)
```
注意函数名与m文件名要一致，每个m文件只含有一个主函数，可有若干子函数辅助主函数（包含在主函数内部）
#6.数据导入导出
保存工作区：save命令
导入工作区：load命令 load xxx.mat
导入JSON ：json2data=loadjson('xxx.json')
#7快捷键
【Ctrl+C】在命令窗口输入，使得运行的程序停下来
【Ctrl+R】注释（对多行有效）
【Ctrl+T】去掉注释（对多行有效）
【F5】      运行程序
【选中代码+F9】执行选中的代码
