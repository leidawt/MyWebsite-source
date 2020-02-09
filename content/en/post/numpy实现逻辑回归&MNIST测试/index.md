---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "numpy实现逻辑回归&MNIST测试"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-12-12T12:00:00+08:00
lastmod: 2018-12-12T12:00:00+08:00
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
### 简述
逻辑回归虽简单，但其思想颇有价值 
Logistic Regression 更原本的翻译应为对数回归，因同义词的缘故，习惯性的称之为逻辑回归了。虽称其为回归，但实际上这个算法是处理分类问题的。
### why Logistic Regression?
我们也可用线性模型进行分类预测，比如设y=wx+b 并判定y>阈值为正类。
{{<figure src = "0.png" title = "" lightbox = "true">}}
但是问题在于，若数据集中有很正类的样本，会极大的影响w权重向自身倾斜，这是不希望看到的，如下
{{<figure src = "1.png" title = "" lightbox = "true">}}
所以，思路便是引入非线性，对很正类的样本加以限制，这里用的是sigmoid函数，利用了其饱和特性。sigmoid函数最初来自18实际对人口增长的研究。
{{<figure src = "2.png" title = "" lightbox = "true">}}
引入后变为这样
{{<figure src = "3.png" title = "" lightbox = "true">}}
实际上，这便是一个采用sigmoid激活函数的单层神经网络
{{<figure src = "4.png" title = "" lightbox = "true">}}
由于非线性的引入，整个函数不再是凸函数了，优化就只有进行迭代求解了。
我们定义代价函数以便进行参数训练。通常有均方误差和交叉熵（极大似然）两种定义，下面分别叙述
1. 均方误差
{{<figure src = "5.png" title = "" lightbox = "true">}}
很朴素的想法，对数据集内每个样本点预测误差求和
2. 交叉熵（极大似然）
交叉熵 与 极大似然 是两种截然不同的思路，但最终的形式是一致的，有些殊途同归的感觉。
交叉熵优化的其实是其KL散度部分，其理解详见
https://www.jianshu.com/p/43318a3dc715
KL散度用于表达两个分布之间相似程度。其其思想简言之就是分布里概率越重的点越要像。
交叉熵误差在分类问题效果远好于均方误差，我们易从概率论的极大似然方法推出其形式
{{<figure src = "6.png" title = "" lightbox = "true">}}
即预测函数取组参数w,b是，其在数据集上能正确表述所有样本点的可能性大小。优化目标自然是找最合适的w,b使得这个概率值最大化。通常还要取一下负对数，一是为了乘法变加法，二是从求max变成求min。
Why better？
原因可由其求导结果体现出来。
https://blog.csdn.net/liweibin1994/article/details/79510237
简言之，均方误差的问题在于其对误差很大的点反而矫正意愿不强烈，如图（这个图来源于著名的Understanding the difﬁculty of training deep feedforward neural networks一文）
{{<figure src = "7.png" title = "" lightbox = "true">}}
### Toy set上实战
使用纯numpy实现

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
df = pd.read_csv('./ex2data1.txt')
m = len(df)  # 数据总数
pos = df[(df['y'] == 1)]
neg = df[(df['y'] == 0)]
x = np.mat(df.iloc[:, 0:-1])
y = np.mat(df.iloc[:, -1]).T  # m*1 mat
X = np.hstack((np.ones((m, 1)), x))  # m*n mat
#theta = np.zeros((X.shape[1], 1))  # n*1 mat
theta=np.array([0,0,0]).reshape(3,1)
cost_history = []
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def J(theta,X,y,mylambda=0):
    theta=theta.reshape(3,1)
    #print(theta.shape,X.shape,y.shape)
    return -1*float((y.T*np.log(h(X,theta))+(1-y.T)*np.log(1-h(X,theta)))/m)+float(theta.T.dot(theta))*mylambda/(2*m)

def h(X,theta):
    return sigmoid(X.dot(theta))

from scipy import optimize

def opt():
    global theta
    result = optimize.fmin(J, x0=theta, args=(X, y,0.), maxiter=400, full_output=True)
    theta=result[0]
    print(theta)
def plot():
        fig = plt.figure()

        plt.plot(pos.iloc[:, 0], pos.iloc[:, 1], 'k+', label='pos')
        plt.plot(neg.iloc[:, 0], neg.iloc[:, 1], 'yo', label='neg')
        # 画边界
        x0 = [float(df.x0.min()), float(df.x0.max())]
        x1 = [float(-1*theta[0]/theta[2] -
                    theta[1]/theta[2]*each) for each in x0]
        plt.plot(x0, x1)
        plt.grid(True)
        plt.show()
opt()
plot()
```
注意这里的优化调用了scipy 的现成算法，会快一些。数据见文末。若想获取非线性决策边界，一个简单的方法是对输入进行扩展。即x->[x,x某平方,交叉项...]然后一并喂给逻辑回归即可。
{{<figure src = "8.png" title = "" lightbox = "true">}}
### 二类分类->n类分类
多分类实现有两种常见思路，一是实现n个二分类器（集成学习的思想，打群架），二是把sigmoid函数换成其多分类版本softmax，其吐出的是对各个类的可能性数值，同时配合交叉熵为代价函数
{{<figure src = "9.png" title = "" lightbox = "true">}}
softmax实际的实现有一些trick，主要是保证其计算稳定而不会溢出。核心方法是都剪掉max(x)再计算exp使得不会溢出。
此外，对于为什么使用softmax而不是其他函数的讨论可见：https://www.zhihu.com/question/40403377
主要是 softmax+交叉熵的求导结果及其简洁
### MNIST测试
这里采用对每个数字分别训练一个二分类器的方法，当时选了均方误差Loss，故训练困难，之后采用了L-BFGS-B优化器代替手写的简单梯度下降进行训练。
**关于L-BFGS-B算法**
http://www.hankcs.com/ml/l-bfgs.html
https://www.zhihu.com/question/46441403
通常的多层神经网络或一些复杂的模型均采用梯度下降及其各种改进方法。不会使用BFGS之类的拟牛顿法，主要有下面考虑
神经网络优化问题有三个特点：大数据（样本多），高维问题（参数极多），非凸，BFGS这类利用二阶导信息的方法中计算成本高，易陷入鞍点的问题就比较显著了。
牛顿和拟牛顿法在凸优化情形下，如果迭代点离全局最优很近时，收敛速率快于gd。
**code**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import random #To pick random images to display
import scipy
from scipy import optimize
from scipy.optimize import minimize
from scipy.special import expit #Vectorized sigmoid function

datafile = './ex3data1.mat'
mat = scipy.io.loadmat( datafile )
X, y = mat['X'], mat['y']
y[y==10]=0 #原数据0 标记为10 这里改回0
#Insert a column of 1's to X as usual
X = np.insert(X,0,1,axis=1)# 插入1

m = X.shape[0]  # 数据总数
n = X.shape[1] #特征数，包括常数项
theta=np.zeros((n,1))
#theta=np.random.random(n)
print(X.shape,y.shape,theta.shape,m,n)
print(np.unique(y))

def J(mytheta,myX,myy,mylambda=0):
    #误差函数，返回float
    mytheta=mytheta.reshape(n,1)#这个处理防止相乘报错。因为输入可能为(n,)而不是（n,1）
    #print('J input theta x y',theta.shape,X.shape,y.shape)
    a=myy.T.dot(np.log(h(mytheta,myX)))
    b=(1-myy).T.dot(np.log(1-h(mytheta,myX)))
    c=theta.T.dot(mytheta)*mylambda/(2*m)
    #print(a.shape,b.shape,c.shape)
    return float(-1/m*(a+b)+c)

def h(mytheta,myX): #Logistic hypothesis function
    #假设函数 返回（m,1）
    res=expit(myX.dot(mytheta)).reshape(m,1)
    #print('h theta X return',mytheta.shape,myX.shape,res.shape)
    return res

def grad(mytheta,myX,myy,mylambda = 0.):
    #计算梯度向量，未考虑正则项 返回（n,）
    #print('grad theta X y',mytheta.shape,myX.shape,myy.shape)
    derta=h(mytheta,myX)-myy
    #print(derta.shape)
    res=(myX.T.dot(derta)).reshape(n)/m
    return res

def opt(mytheta,myX,myy,c,mylambda=0.):
    #递归优化 c为类名（0-9）
    #minimize的输入mytheta为(n,)其传给调用函数的形式也为(n,)
    #这里传入(myy == c)*1是一个[0 0 0 1 1 ... ]向量
    #maxiter最大迭代数，disp显示优化信息
    #由于函数的复杂性，其不一定收敛的彻底，50次迭代已经很好，随仍可继续下降，但可能过拟合
    #L-BFGS-B为优化的拟牛顿法，因为利用了二阶倒数信息，下降比较快。不传入自定义的grad会导致
    #其使用数值方法计算grad，及其缓慢
    res = minimize(J, mytheta, args=( myX, (myy == c)*1), method='L-BFGS-B',
                       jac=grad, options={'maxiter':50,'disp':False})
    #print(res)
    #res.x 是优化好的theta值
    return res.x


#final_theta=opt(theta,X,y,1)
#print(final_theta.shape)
#print(X[495:505,:].dot(final_theta))
#print(y[495:505,:])
#对0-9分别训练，以预测值最大的类别作为最终预测
thetas=np.zeros((10,n))#(10,401)
for each in range(10):
    thetas[each]=opt(theta,X,y,each)
predict=thetas.dot(X.T)# (10,5000)
#argmax函数对每一类求最大值index
predict=predict.argmax(axis=0) #(5000,)
#最后统计正确率
print(np.mean((predict==y.reshape(m)))*100,'%')
#产生的RuntimeWarning可能来自某些变量在优化中变0，导致log计算出错，会变为NaN，不影响最终结果
```
在我的低压cpu上6.33 s 就可在训练集获得96.84 %的正确率，可见表达力还是不错的，一是采用10个分类器的原因，二是MNIST本身确实简单
### 参考资料
cs229
李宏毅机器学习
### DATA
链接: https://pan.baidu.com/s/1rxQM1HCICnAaUW3ih2GeYQ 提取码: ryj4
