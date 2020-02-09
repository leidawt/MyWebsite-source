---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Bayesian Methods for Hackers笔记"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2019-05-09T12:00:00+08:00
lastmod: 2019-05-09T12:00:00+08:00
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
这本开源书从实践角度初步入门概率编程。值得学习的有：
1.大佬优秀的可视化技巧
2.TFP包基础
3.概率编程和贝叶斯思想

书包含使用不同框架的版本，这里用TFP的版本
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
# 一些util函数
此段作者定义了一些实用的函数并配置了绘图风格，应当首先执行
```python
#@title Imports and Global Variables (run this cell first)  { display-mode: "form" }
"""
The book uses a custom matplotlibrc file, which provides the unique styles for
matplotlib plots. If executing this book, and you wish to use the book's
styling, provided are two options:
    1. Overwrite your own matplotlibrc file with the rc-file provided in the
       book's styles/ dir. See http://matplotlib.org/users/customizing.html
    2. Also in the styles is  bmh_matplotlibrc.json file. This can be used to
       update the styles in only this notebook. Try running the following code:

        import json
        s = json.load(open("../styles/bmh_matplotlibrc.json"))
        matplotlib.rcParams.update(s)
"""
#!pip3 install -q wget
from __future__ import absolute_import, division, print_function
#@markdown This sets the warning status (default is `ignore`, since this notebook runs correctly)
warning_status = "ignore" #@param ["ignore", "always", "module", "once", "default", "error"]
import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status, category=DeprecationWarning)
    warnings.filterwarnings(warning_status, category=UserWarning)

import numpy as np
import os
#@markdown This sets the styles of the plotting (default is styled like plots from [FiveThirtyeight.com](https://fivethirtyeight.com/))
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt
plt.style.use(matplotlib_style)
import matplotlib.axes as axes
from matplotlib.patches import Ellipse
%matplotlib inline
import seaborn as sns; sns.set_context('notebook')
from IPython.core.pylabtools import figsize
#@markdown This sets the resolution of the plot outputs (`retina` is the highest resolution)
notebook_screen_res = 'retina' #@param ['retina', 'png', 'jpeg', 'svg', 'pdf']
%config InlineBackend.figure_format = notebook_screen_res

import tensorflow as tf
tfe = tf.contrib.eager

# Eager Execution
#@markdown Check the box below if you want to use [Eager Execution](https://www.tensorflow.org/guide/eager)
#@markdown Eager execution provides An intuitive interface, Easier debugging, and a control flow comparable to Numpy. You can read more about it on the [Google AI Blog](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html)
use_tf_eager = False #@param {type:"boolean"}

# Use try/except so we can easily re-execute the whole notebook.
if use_tf_eager:
    try:
        tf.enable_eager_execution()
    except:
        pass

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

  
def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
        `namedtuple` or combinations thereof.

    Returns:
        ndarrays: Object with same structure as `tensors` except with `Tensor` or
          `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def session_options(enable_gpu_ram_resizing=True, enable_xla=True):
    """
    Allowing the notebook to make use of GPUs if they're available.
    
    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
    algebra that optimizes TensorFlow computations.
    """
    config = tf.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        # `allow_growth=True` makes it possible to connect multiple colabs to your
        # GPU. Otherwise the colab malloc's all GPU ram.
        config.gpu_options.allow_growth = True
    if enable_xla:
        # Enable on XLA. https://www.tensorflow.org/performance/xla/.
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
    return config


def reset_sess(config=None):
    """
    Convenience function to create the TF graph & session or reset them.
    """
    if config is None:
        config = session_options()
    global sess
    tf.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession(config=config)

reset_sess()
```
其中reset_sess()和evaluate()尤为关键，封装了sess操作便于直接得到结果
例：

```python
'''
evaluate函数使用方法（其将tf_tensor执行session以转换到numpy格式）
注：采用以下的命名约定减小混乱：下划线结尾的变量表示转换到numpy的版本
'''
reset_sess()#构建或重启全局session

parameter = tfd.Exponential(rate=1., name="poisson_param").sample()#从指数分布采样一个值（tensor）

[ parameter_ ] = evaluate([ parameter ])#转换

print("Sample from exponential distribution before evaluation: ", parameter)
print("Evaluated sample from exponential distribution: ", parameter_)
```
Sample from exponential distribution before evaluation:  Tensor("poisson_param/sample/Reshape:0", shape=(), dtype=float32)
Evaluated sample from exponential distribution:  0.3230632
# Chapter 1 引言
概率领域有两大学派：频率派和贝叶斯派，其对概率的观点有所不同
**频率派**视概率是事件的长期频率。这在解释多次事件（如飞机坠毁）在中是没有问题的，但处理单次事件（如xxx竞选成功的概率）会面临问题。
**贝叶斯派**视概率为对事件发生信心的度量。这种观点解释单次事件就很直观和清晰了（如xxx竞选成功的概率就是对xxx成功竞选的信心）。
故实际上贝叶斯派对概率的看法是很符合人最初的直觉的，频率派的思想来自后期训练。
通常的，记我们对某事件的信度为P(A)，称先验概率（prior probability），观测到一些证据X后，看法会改变，形成P(A|X)，称后验概率（posterior probability）。有时候先验可能很错误或不包含什么有用的东西，不过通过观测证据的修正，我们得到的后验会变得更加正确。
贝叶斯推断和频率推断回答问题的方式有很大的不同，以bug测试问题举例说明：
1.频率派
问：我的代码通过了所有的X测试;我的代码没有bug吗?
答：没错误
2.贝叶斯派
问：我的代码经常有错，我的代码通过了所有的X测试;我的代码没有bug吗?
答：有80%的可能性没错误
这里有两大区别：一是后者回答可能性，前者回答估计结果。二是贝叶斯派还在体温在给出了可选的先验，让贝叶斯推断同时考虑一下“我的代码经常有错”这个先验条件。此处贝叶斯的优势在于其保持了在给出的证据不充分时回答的不确定性。
**贝叶斯公式：**
{{<figure src = "0.png" title = "" lightbox = "true">}}
这个简单的式子是贝叶斯推理的核心，其重要意义是构建了先验和后验之间的关系。
其中P(X|A)又称先验A的似然度，也写作L(A|X)。分母实则是个归一化常数，所以贝叶斯公式也经常写作$P(A|X)\propto P(X|A)P(A)$。
**贝叶斯推理简单实例**
现在要从下图所示数据中分析出来人们的编码习惯是否与时间相关。可以注意到贝叶斯推理将参数视作随机变量的观点。
{{<figure src = "1.png" title = "" lightbox = "true">}}
鉴于泊松分布适合于描述单位时间内随机事件发生的次数的概率分布，如某一服务设施在一定时间内受到的服务请求的次数，电话交换机接到呼叫的次数、汽车站台的候客人数、机器出现的故障数。故可以记第i天消息数 $C_i \sim \text{Poisson}(\lambda)$。这里参数 $\lambda$ 代表独立实事件的发生概率，是未知的。
可以从数据观察到后期消息数明显有提高，因此考虑如下假设：
$$\lambda = 
\begin{cases} \lambda_1  ; \text{if } t \lt \tau \cr
\lambda_2 ; \text{if } t \ge \tau
\end{cases}
$$
即认为$\lambda$是时变的。继续为 $\lambda_1$ ，$\lambda_2$建模。考虑到其连续性，使用了指数分布，$\alpha$作为超参数（超参数准确含义是影响其他参数的参数）。
$$
\lambda_1 \sim \text{Exp}( \alpha ) \\
\lambda_2 \sim \text{Exp}( \alpha )
$$
最后由于不知道$\tau$的任何先验知识，只好以均匀分布建模。即$\tau \sim \text{DiscreteUniform(1,70) }$。至此模型假设已经全部定义完毕（通过指定先验分布）。接下来是求解，由于较为复杂，使用概率编程库来计算之。
**概率编程库--TensorFlow Probability**
作为TensorFlow 的子项目，TFP的核心优势是数据计算性能的高度优化。TFP建模的关键在于定义描述模型结构的joint_log_prob函数，该函数以数据和欲求的模型参数为输入，返回联合概率的对数（即返回参数在数据中的似然度），这个似然度会用于后续采样算法。
上面定义的模型归纳如下：
{{<figure src = "2.png" title = "" lightbox = "true">}}
{{<figure src = "3.png" title = "" lightbox = "true">}}
joint_log_prob定义为：
```python
def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    tfd = tfp.distributions
    #指定超参数
    alpha = np.array(1. / count_data.mean(), np.float32)
    
    #指明参数的分布函数
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)
    rv_tau = tfd.Uniform()
    lambda_ = tf.gather(
         [lambda_1, lambda_2],
         indices=tf.to_int32(tau * count_data.size <= np.arange(count_data.size)))
    rv_observation = tfd.Poisson(rate=lambda_)
    
 	# 返回joint prob 因为取了log，故乘变加
    return (
         rv_lambda_1.log_prob(lambda_1)
         + rv_lambda_2.log_prob(lambda_2)
         + rv_tau.log_prob(tau)
         + tf.reduce_sum(rv_observation.log_prob(count_data))
    )
```
然后经过MCMC采样算法（后续章节说明）的迭代计算，欲求参数的后验分布就能得到了，如下所示：
{{<figure src = "4.png" title = "" lightbox = "true">}}
可以观察到后验分布的结果保留了不确定性。
# Chapter 2 TFP包
TensorFlow新版中加入了eager模式增加了对立即计算的支持。此书支持eager和非eager的模式，统一使用evaluate函数得到结果。
## 分布函数
tfp.distributions包含大量常用分布类。https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
这些分布类的关键概念是：
Event shape：多变量概率分布的维数，5维就是[5]
Batch shape：独立不同分布的个数
建立随机变量可用

```python
var= tfd.Exponential(rate=1., name="poisson_param")
```

建立确定性变量可用

```python
new_deterministic_variable = tfd.Deterministic(name="deterministic_variable", 
                                               loc=var.sample())
```
所谓确定性变量既是取值固定的随机变量，为了抽象和统一接口方便

## 贝叶斯A/B测试
这是一种类似本科概率论中假设检验的方法。下面以一个例子说明。
例：假设$P_A$表示某购物网站的 浏览->购买 比例，$P_B$表示另一购物网站的 浏览->购买 比例。现在希望从在两个网站观察到的 购买数 来估计真实概率$P_A$，$P_B$ 并推测谁更高。

可以通过贝叶斯推理来估计$P_A$,$P_B$
首先制作人工数据。

```python
reset_sess()

#set constants
prob_true = 0.05  # 指定P_A
N = 1500          # 总浏览数（采样数）

# 以 Ber(0.05) 为采样分布
occurrences = tfd.Bernoulli(probs=prob_true).sample(sample_shape=N, seed=10)
print(evaluate(occurrences))
```
造出数据后可用joint_log_prob来表达数据和参数的匹配程度，如下：

```python
def joint_log_prob(data,A_guess):
    rv_A_guess=tfd.Uniform(low=0.,high=1.)
    rv_data=tfd.Bernoulli(probs=A_guess)
    return (
        rv_A_guess.log_prob(A_guess)+
        tf.reduce_sum(rv_data.log_prob(data))
    )
```
之后通过MCMC采样来从造出来的数据估计真实$P_A$。
MCMC采样方法不停尝试不同的A_guess，并调用joint_log_prob判断猜测的好坏最终得到$P_A$的后验估计。如下：
{{<figure src = "5.png" title = "" lightbox = "true">}}
可用看到中心在真实值0.05处，拖尾是由于不确定造成的。$P_B$的处理方法相同。
令$delta=P_A-P_B$。通过MCMC一并得到他们的后验估计结果：
{{<figure src = "6.png" title = "" lightbox = "true">}}
注：由于设置了B站的N为A的一半，故B的拖尾更大一些。

最后，求np.mean(delta> 0))即可得到如下的推理结果：有0.706的可能A好于B。这便是A/B实验的基本方法。

## 实例：挑战者号爆炸
数据如下：
{{<figure src = "7.png" title = "" lightbox = "true">}}
数据描述o型圈失效与温度的关系。
观察其形状，可考虑带偏置的logistic 函数描述数据：
$$p(t) = \frac{1}{ 1 + e^{ \;\beta t + \alpha } } $$
接下来开始建模，目标是确定 $\beta, \alpha$ 。
可以用正态分布$N( \mu, 1/\tau)$ 作为 $\beta, \alpha$的先验分布。（原作者选了N(0,1/1000)，个人认为此次使用uniform（-100,100）可能更有效，N(0,1/1000)引入的先验过多且无依据）。
最后的数据预测可使用伯努利分布来做$$ \text{Defect Incident, }D_i \sim \text{Ber}( \;p(t_i)\; ), \;\; i=1..N$$
综上，失效与否是以时变的伯努利分布确定的，其参数是以带偏置的logistic 函数确定的，logistic 的参数是求解目标，且以N(0,1/1000)为先验。
之后使用MCMC求解即可。下图画出了由 $\beta, \alpha$的后验确定的logistic函数。紫色区域是$\beta, \alpha$的后验分布的95%置信区间。
{{<figure src = "8.png" title = "" lightbox = "true">}}
坠毁当天的t=31，用的到的推理结果进行预测，结果显示出问题几乎是必然的：

```python

plt.figure(figsize(12.5, 3))

prob_31 = logistic(31, posterior_beta_, posterior_alpha_)

[ prob_31_ ] = evaluate([ prob_31 ])

plt.xlim(0.98, 1)
plt.hist(prob_31_, bins=10, density=True, histtype='stepfilled')
plt.title("Posterior distribution of probability of defect, given $t = 31$")
plt.xlabel("probability of defect occurring in O-ring");
```
{{<figure src = "9.png" title = "" lightbox = "true">}}
一个十分值得关注的问题是如何评价模型十分很适合数据？一个不错的方法是从模型采样人工数据并与真实数据对比。接下来作者提出了一种可视化这种对比的不错的新方法。
首先从模型求一下对应数据点的后验（左为后验，右为真实数据）
posterior prob of defect | realized defect 
0.47                     |   0
0.20                     |   1
0.25                     |   0
0.31                     |   0
...
之后按后验概率进行排序
probb | defect 
0.01  |   0
0.01  |   0
0.02  |   0
0.03  |   0
...
最后调用如下的函数绘制
```python
import matplotlib.pyplot as plt

def separation_plot( p, y, **kwargs ):
    """
    This function creates a separation plot for logistic and probit classification. 
    See http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf
    
    p: The proportions/probabilities, can be a nxM matrix which represents M models.
    y: the 0-1 response variables.
    
    """    
    assert p.shape[0] == y.shape[0], "p.shape[0] != y.shape[0]"
    n = p.shape[0]

    try:
        M = p.shape[1]
    except:
        p = p.reshape( n, 1 )
        M = p.shape[1]

    colors_bmh = np.array( ["#eeeeee", "#348ABD"] )


    fig = plt.figure( )
    
    for i in range(M):
        ax = fig.add_subplot(M, 1, i+1)
        ix = np.argsort( p[:,i] )
        #plot the different bars
        bars = ax.bar( np.arange(n), np.ones(n), width=1.,
                color = colors_bmh[ y[ix].astype(int) ], 
                edgecolor = 'none')
        ax.plot( np.arange(n+1), np.append(p[ix,i], p[ix,i][-1]), "k",
                 linewidth = 1.,drawstyle="steps-post" )
        #create expected value bar.
        ax.vlines( [(1-p[ix,i]).sum()], [0], [1] )
        plt.xlim( 0, n)
        
    plt.tight_layout()
    
    return

plt.figure(figsize(11., 3))
separation_plot(posterior_probability_, D_)
```
结果如下：
{{<figure src = "10.png" title = "" lightbox = "true">}}
蛇形线表示排序后的概率，蓝色条表示缺陷，而空白表示非缺陷。随着概率的增加，我们看到越来越多的缺陷发生。在右侧，图中显示，当后验概率较大(直线接近1)时，缺陷会更多。这是良好的行为。理想情况下，所有的蓝条都应该靠近右边，偏离右边反映了预测失误。
使用这样的图标可以比较直观的评价模型，如：
{{<figure src = "11.png" title = "" lightbox = "true">}}
{{<figure src = "12.png" title = "" lightbox = "true">}}
# Chapter 3 MCMC
马尔科夫链蒙特卡洛采样（MCMC）方法是用来在概率空间，通过随机采样估算兴趣参数的后验分布的。https://zhuanlan.zhihu.com/p/37121528。
其余的采用方法还有拉普拉斯近似和VAE等。

MCMC的思路大致如下：
1.从当前位置开始。 
2.考虑移动到一个新的位置(探索附近的点)。 
3.根据位置对数据和先验分布的遵从程度接受/拒绝新位置。 
4.如果接受了，那就移动到新位置。回到步骤1。 否则:不移动到新位置。回到步骤1。
5. 在大量迭代之后，返回所有接受的位置。

一个使用混合模型的无监督聚类例子
目标：对形如下图的双峰数据进行无监督聚类。
{{<figure src = "13.png" title = "" lightbox = "true">}}
拟使用的混合模型是$P(x|\theta) = \sum_{k=1}^{K}{\alpha_{k}\phi(x|\theta_{k})}$。即k个分布的加权相加。当每个分布是高斯分布时候，就是常用的gmm（高斯混合模型）。
先制作形如上图的人工数据，方法如下：
设点属于$C_1$类的概率为p，$C_2$类的1-p，$p \sim U(0,1)$。
设$\sigma_1,\sigma_2\sim U(0,100);\mu_0\sim N(120,10);\mu_1\sim N(190,10)$
其中120,190是目视观察的估计值。
模型的待求参数是p和两个高斯分布的均值方差

此模型的联合概率函数定义如下，其中MixtureSameFamily函数是tfp提供加权和的函数，换为$p*N(120,10)+(1-p)*N(190,10)$意思一样。
```python
def joint_log_prob(data_, sample_prob_1, sample_centers, sample_sds):
    """
    Joint log probability optimization function.
        
    Args:
      data: tensor array representation of original data
      sample_prob_1: Scalar representing probability (out of 1.0) of assignment 
        being 0
      sample_sds: 2d vector containing standard deviations for both normal dists
        in model
      sample_centers: 2d vector containing centers for both normal dists in model
    Returns: 
      Joint log probability optimization function.
    """  
    ### Create a mixture of two scalar Gaussians:
    rv_prob = tfd.Uniform(name='rv_prob', low=0., high=1.)
    sample_prob_2 = 1. - sample_prob_1
    rv_assignments = tfd.Categorical(probs=tf.stack([sample_prob_1, sample_prob_2]))
    
    rv_sds = tfd.Uniform(name="rv_sds", low=[0., 0.], high=[100., 100.])
    rv_centers = tfd.Normal(name="rv_centers", loc=[120., 190.], scale=[10., 10.])
    
    rv_observations = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Normal(
          loc=sample_centers,       # One for each component.
          scale=sample_sds))        # And same here.
    return (
        rv_prob.log_prob(sample_prob_1)
        + rv_prob.log_prob(sample_prob_2)
        + tf.reduce_sum(rv_observations.log_prob(data_))      # Sum over samples.
        + tf.reduce_sum(rv_centers.log_prob(sample_centers)) # Sum over components.
        + tf.reduce_sum(rv_sds.log_prob(sample_sds))         # Sum over components.
    )
    sum_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
    # Note: for easy debugging, uncomment the following:
    return sum_log_prob
```
接下来是重要的MCMC部分：

```python
number_of_steps=25000 #采用次数
burnin=1000 #前期丢弃样本数（因前期不准）

# Set the chain's start state.
initial_chain_state = [
    tf.constant(0.5, name='init_probs'),
    tf.constant([120., 190.], name='init_centers'),
    tf.constant([10., 10.], name='init_sds')
]

# Since MCMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Identity(),       # Maps R to R.
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(data_, *args)


# Initialize the step_size. (It will be automatically adapted.)
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.5, dtype=tf.float32),
        trainable=False,
        use_resource=True
    )


# Defining the HMC
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

# Sample from the chain.
[
    posterior_prob,
    posterior_centers,
    posterior_sds
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)

# Initialize any created variables.
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

evaluate(init_g)
evaluate(init_l)
[
    posterior_prob_,
    posterior_centers_,
    posterior_sds_,
    kernel_results_
] = evaluate([
    posterior_prob,
    posterior_centers,
    posterior_sds,
    kernel_results
])
    
new_step_size_initializer_ = kernel_results_.inner_results.is_accepted.mean()
print("acceptance rate: {}".format(
    new_step_size_initializer_))
new_step_size_initializer_
print("final step size: {}".format(
    kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))
```
最后结果如下（两个分布的参数是取的mcmc给出的后验分布的均值）：
{{<figure src = "14.png" title = "" lightbox = "true">}}
最终，为了判断新的样本点应该属于哪一个类，可以使用贝叶斯公式将
$P(L_x = 1| x = 175 ) \gt P(L_x = 0| x = 175 )$ 转为
$P( x=175  | L_x = 1  )P( L_x = 1 ) \gt  P( x=175  | L_x = 0  )P( L_x = 0 )$
最后一个小问题：如何判断收敛？
可以考虑相关性。
**相关系数**是一个数，定义为两个变量之间的协方差和标准差的商（这里是皮尔逊相关系数）：
$${\displaystyle \rho _{X,Y}={\mathrm {cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={E[(X-\mu _{X})(Y-\mu _{Y})] \over \sigma _{X}\sigma _{Y}}}$$
**互相关cross-correlation**是一个算子，描述两个序列相似程度，f,g之间的互相关以$f\star g$表示。$f\star g = \overline{f(-t)}\ast g(t)$ 上划线表共轭。故其与卷积很类似，当只考虑实数时，卷积就是$f(t)\star g(-t)$。
**自相关Autocorrelation**即$f\star f$。 在统计量中，一个随机过程的自相关是该过程在不同时间的值之间的皮尔逊相关，作为两个时间间隔或时滞的函数。${\displaystyle \operatorname {R} _{XX}(t_{1},t_{2})=\operatorname {E} [X_{t_{1}}{\overline {X_{t_{2}}}}]} {\displaystyle }$。具体的：
http://nanshu.wang/post/2015-03-15/
{{<figure src = "15.png" title = "" lightbox = "true">}}
pytho中计算1-n阶自相关的方法是：
```python
def autocorr(x):
    # from http://tinyurl.com/afz57c4
    result = np.correlate(x, x, mode='full')
    result = result / np.max(result)
    return result[result.size // 2:]
```
例如对两个随机过程$$x_t \sim \text{Normal}(0,1), \;\; x_0 = 0$$$$y_t \sim \text{Normal}(y_{t-1}, 1 ), \;\; y_0 = 0$$
其自相关如下：
{{<figure src = "16.png" title = "" lightbox = "true">}}
可以看到时间间隔越远相关性越差。MCMC算法的特点是探索附近的值，其返回的点与前一次的位置相关。因此当MCMC链开始呈现很高的自相关时，表明不再很好的探索周边区域。（MCMC的收敛是指收敛到一个区域，而非通常意义上收敛到某一点）。低自相关不是收敛的必要条件，但它是充分的。TFP有一个内置的自相关工具。
# Chapter 4 
大数定律：
$$\frac{1}{N} \sum_{i=1}^N Z_i \rightarrow E[ Z ],  \;\;\; N \rightarrow \infty.$$
随着N的增长，收敛会放缓，其斜率约为$\frac{ \sqrt{ \; Var(Z) \; } }{\sqrt{N} }$：
{{<figure src = "17.png" title = "" lightbox = "true">}}
在贝叶斯推理中，要特别主要N不大时大数定律的失效。
**例子：Kaggle的美国人口普查回复率挑战**
任务目标是利用人口普查变量(收入中位数、社区女性人数、拖车停车场数量、儿童平均人数等)预测一个社区的人口普查信件回复率，其测量值在0到100之间。数据是区块给出的（每社区人口数不同，约几千），原始数据如下：
{{<figure src = "18.png" title = "" lightbox = "true">}}
可以观察到呈现典型的三角形，随着区块样本量增大，回复率方差收紧，这是大数定律的典型体现。
**例子：如何给reddit帖子排序**
给出了按照赞踩比进行帖子排序的方法。真实的赞踩比是个隐变量，可以对其进行推理。直接使用观测的赞踩比作为估计量并不是好主意，其未考虑样本量的问题而违反大数定律。使用如下的模型：$p\sim U(0,1),observations\sim Binomial(p,N)$。爬取了一些帖子，使用MCMC采样出赞踩比的后验。下图虚线是95%置信下限。
也有解析的解法，快得多：
$$ \frac{a}{a + b} - 1.65\sqrt{ \frac{ab}{ (a+b)^2(a + b +1 ) } }$$
其中$$
 a = 1 + u \\
 b = 1 + d \\
 u=赞数 \\
 d=踩数 \\
$$
{{<figure src = "19.png" title = "" lightbox = "true">}}
这样充分考虑了样本量的问题。不考虑样本量和试图对不稳定的对象进行排序会导致病态的排序。
# chapter 5 损失函数
一些损失函数设计的例子：
简单的均方损失函数：$L( \theta, \hat{\theta} ) = f( \theta, \hat{\theta} )$，对过大的错误惩罚很大。
不平衡均方损失函数，可进行偏好性惩罚：
$$ L( \theta, \hat{\theta} ) = \begin{cases} ( \theta -  \hat{\theta} )^2  \hat{\theta} \lt \theta \\\\ c( \theta -  \hat{\theta} )^2  \hat{\theta} \ge \theta, \;\; 0\lt c \lt 1 \end{cases}$$
绝对值损失函数，对大偏差的惩罚不那么极端：$L( \theta, \hat{\theta} ) = | \theta -  \hat{\theta} |$
0-1损失函数，常用于分类任务：$L( \theta, \hat{\theta} ) = \mathbb{1}_{ \hat{\theta} \neq \theta }$
log损失函数，常用于分类：$L( \theta, \hat{\theta} ) = -\theta\log( \hat{\theta} ) - (1- \theta)\log( 1 - \hat{\theta} ), \; \; \theta \in {0,1}, \; \hat{\theta} \in [0,1]$
改进的绝对误差：$L( \theta, \hat{\theta} ) = \frac{ | \theta - \hat{\theta} | }{ \theta(1-\theta) }, \; \; \hat{\theta}, \theta \in [0,1]$ 这个很有意思，强调对接近0和1的真值的预测。
指数损失函数：$L( \theta, \hat{\theta} ) =  1 - \exp \left( -(\theta -  \hat{\theta} )^2 \right)$ ，其值在（0,1）区间，大误差时饱和，故对很大的预测偏离不很在意。
{{<figure src = "20.png" title = "" lightbox = "true">}}
在贝叶斯推断中，损失函数定义在分布上，而非分布的某些采用值：
$$ l(\hat{\theta} ) = E_{\theta}\left[ \; L(\theta, \hat{\theta}) \; \right] $$
根据大数定律：
$$\frac{1}{N} \sum_{i=1}^N \;L(\theta_i, \hat{\theta} ) \approx E_{\theta}\left[ \; L(\theta, \hat{\theta}) \; \right]  = l(\hat{\theta} ) $$
根据任务目标合理设计loss很重要。
**例子：金融预测**
人工数据集如下：
$X\sim N(0,1)*0.025,Y\sim 0.5X+0.01N(0,1)$,采样100点
{{<figure src = "21.png" title = "" lightbox = "true">}}
模型定义：$R = \alpha + \beta x + \epsilon$，参数先验均为N(0,1)
通过MCMC采样得到三个参数的后验分布。下面定义一个stock_loss并以其为依据给出最合适的预测。
stock_loss形如：

```python
def stock_loss(true_return, yhat, alpha=100.):
    """
    Stock Loss function
    
    Args:
      true_return: float32 Tensor representing the true stock return
      yhat: float32
      alpha:float32
      
    Returns:
      float: absolute value of the difference
      between `true_return` and `yhat`
    """
    if true_return * yhat < 0:
        # opposite signs, not good
        return alpha * yhat ** 2 - tf.sign(true_return) * yhat \
            + tf.abs(true_return)
    else:
        return tf.abs(true_return - yhat)
```
接下来是简单的优化问题：
$$R_i(x) =  \alpha_i + \beta_ix + \epsilon \\ \arg \min_{r} \;\;E_{R(x)}\left[ \; L(R(x), r) \; \right] $$
此处R(x)是一个分布（因，$\alpha$等都是分布，故就是个随机变量函数的分布。编程上就是一堆采样值），r是预测（一个标量的数），L是上面的stock_loss。目标是找使得期望loss最小的预测r。具体argmin是用scipy的fmin做的，其实现了奈德-米德单纯形算法。
最终的结果如下：
{{<figure src = "22.png" title = "" lightbox = "true">}}
可以看出，在stock_loss下做出的预测并不是绝对精度最优，而对stock_loss最优。在x=0附近，因为采样结果很不明确，倾向于给出接近0的预测。作者称之为稀疏的预测，在不很确定时干脆就不行动，这在真实的决策过程中是非常合理并符合人类直觉的。

**一个很有有价值的例子：kaggle暗物质挑战赛**
 这是一个由给出的星图来判断暗物质位置的任务。任务数据量较小，百量级，冠军方案使用贝叶斯推理。
 给出的数据形如下。如图是一片天空。其中的蓝色椭圆形小点代表星系，收到暗物质团的巨大质量的影响，时空弯曲，导致原本随机均匀分布的星系的位置和偏心率受到影响。
{{<figure src = "23.png" title = "" lightbox = "true">}}
数据以csv文件给出，一张星图一个文件，每个文件包含了数百个星系的位置和偏心率描述，每张星图有1--3个暗物质（1大2小）。偏心率e1 e2描述如下图，e1描述横向拉伸，e2描述45°方向拉伸。
{{<figure src = "24.png" title = "" lightbox = "true">}}
竞赛冠军Tim Salimans使用贝叶斯推理方法进行了如下的建模思路。
以贝叶斯公式为基础，为了获得p(x|e)，即知道星系分布e的情况下暗物质x的分布。应用贝叶斯公式倒转依赖。$p(x|e)\propto p(e|x)p(x)$。因为是x影响e的，故求取p(x|e)要在逻辑上容易得多。p(x)也不难通过假设确定。
具体的，考虑暗物质x的 坐标(x,y) 和 质量m，赋予如下先验
$$
x_i \sim \text{Uniform}( 0, 4200)\\
y_i \sim \text{Uniform}( 0, 4200), \;\; i=1,2,3\\
m_{\text{large} } = \text{Uniform}( 40, 180 ) \\
m_{\text{small} } = 20 \\
$$
下面设定p(e|x)，设
$$ e_i | ( \mathbf{x}, \mathbf{y} ) \sim \text{Normal}( \sum_{j = \text{halo positions} }d_{i,j} m_j f( r_{i,j} ), \sigma^2 ) $$
来描述星系e在受暗物质x影响下的分布。
其中f项体现暗物质距离的影响。
大暗物质:

$$ f( r_{i,j} ) = \frac{1}{\min( r_{i,j}, 240 ) } $$
小暗物质

$$ f( r_{i,j} ) = \frac{1}{\min( r_{i,j}, 70 ) } $$
$r_{i,j}$ 是欧氏距离

d项体现方位的影响。
其计算很奇葩，返回$\sin(2\arctan(\frac{\Delta y}{\Delta x}))$和$\cos(2\arctan(\frac{\Delta y}{\Delta x}))$

m项就是暗物质的质量
方差项直接设为星图的观测偏心率，约0.05

使用星图数据data以MCMC方法处理p(e|x)，得到x 位置和质量的后验分布（先只考虑最大的暗物质）。注：此处的位置后验是以MCMC采样的后验分布的均值和方差绘制的。
如下是后验的绘制及其与真实值对比
{{<figure src = "25.png" title = "" lightbox = "true">}}
推广到3个。（这里直接绘制MCMC给出的后验）
{{<figure src = "26.png" title = "" lightbox = "true">}}
方案最后提交的结果是后验均值，并获得了不错的效果。

方案总结：
1.此方案并未直接使用给定的暗物质真实位置进行监督学习，真实位置只用于评价性能，起到验证集作用。
2.先验分布很精简，可避免过拟合
3.用了很多小trick，比如min(dis,240)钳位操作能有效避免极端值的影响。又比如观察到数据普遍由一个大两个小组成，因此考虑进行了区别对待。
4.此方案体现了精心设计的贝叶斯推断方法在小数据中的强大性能。

# chapter 6 先验
略




