---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "经典Policy Iteration实现"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2019-07-16T12:00:00+08:00
lastmod: 2019-07-16T12:00:00+08:00
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
本文总结了强化学习中的经典Policy Iteration方法，在一个租车问题背景之下使用python实现，踩了一下python多进程的坑。。
主要仿写：
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental_synchronous.py
#  背景问题描述
假设租车公司有两个场地 A,B 最大车辆数20，借出车辆收益10，在两地之间调运车辆收益-2，每天最多移动5辆。假设A,B两地的借还数服从参数分别为3 3 4 2 的泊松分布。问题的目标就是找到最佳的调运方案使得总收益最大。
# Policy Iteration
 Policy Iteration是交替使用policy evaluation和policy improvement的方法
 
首先先定义分布函数，由于调用量大，以lru_cache进行加速

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import functools
import multiprocessing as mp
import itertools
import time

@functools.lru_cache()
def poi(n, lam):
    return poisson.pmf(n, lam)
```
定义问题的常量，其中TRUNCATE = 9表明只考虑泊松分布的截断，以便遍历借还情况
```python
############# PROBLEM SPECIFIC CONSTANTS #######################
MAX_CARS = 20
MAX_MOVE = 5
MOVE_COST = -2

RENT_REWARD = 10
# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3
# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4
# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3
# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2
# 限制泊松分布最大取值，否则会有无限多种state
TRUNCATE = 9
# bellman方程的GAMMA
GAMMA = 0.9
# 并行进程数
MP_PROCESS_NUM = 8
################################################################
```
定义问题的贝尔曼方程，根据给定的参数，遍历借还情况，给出expected_return。这个函数的计算量很大，有O(N^4)

```python
def bellman(values, state, action):
    expected_return = 0
    # 先减掉移动车辆的开销 因为是固定的
    expected_return += MOVE_COST*abs(action)
    for req1, req2 in itertools.product(range(TRUNCATE), range(TRUNCATE)):
        # 遍历两个地区每一组可能的借车请求
        # 按action在两地间移动车辆

        # 确保够移在policy_improvement 中实现

        # 确保不超过两地最多车数限制，多了认为是移动到了别的场地
        num_of_cars_loc1 = int(min(state[0]-action, MAX_CARS))
        num_of_cars_loc2 = int(min(state[1]+action, MAX_CARS))
        # 实际借车数量
        real_rent_loc1 = min(num_of_cars_loc1, req1)
        real_rent_loc2 = min(num_of_cars_loc2, req2)
        num_of_cars_loc1 -= real_rent_loc1
        num_of_cars_loc2 -= real_rent_loc2
        # 借出受益
        reward = (real_rent_loc1+real_rent_loc2)*RENT_REWARD
        # 本state的可能性
        prob = poi(req1, RENTAL_REQUEST_FIRST_LOC) * \
            poi(req2, RENTAL_REQUEST_SECOND_LOC)
        # 还车
        for ret1, ret2 in itertools.product(range(TRUNCATE), range(TRUNCATE)):
            # 按照题目意思，多还不考虑
            num_of_cars_loc1_ = int(min(num_of_cars_loc1+ret1, MAX_CARS))
            num_of_cars_loc2_ = int(min(num_of_cars_loc2+ret2, MAX_CARS))
            prob_ = poi(ret1, RETURNS_FIRST_LOC) * \
                poi(ret2, RETURNS_SECOND_LOC)*prob
            # 计算经典贝尔曼方程，其中prob_就是p(s',r|a,s)其中s'对应
            # (num_of_cars_loc1_,num_of_cars_loc2_)
            expected_return += prob_ * \
                (reward+GAMMA*values[num_of_cars_loc1_, num_of_cars_loc2_])
    return expected_return
```
之后policy evaluation 就很好实现了
采用mutiprocessing来对每个state并行加快速度
注意policy_evaluation_helper需要是global函数，不能在policy_evaluation中定义。另需注意states迭代器每次需要重新做一下，mp.Pool不会自动重置迭代器。
```python
def policy_evaluation_helper(state, values, policy):
    action = policy[state[0], state[1]]
    expected_return = bellman(values, state, action)
    return expected_return, state
def policy_evaluation(values, policy):
    # 并行的遍历更新values表,返回新values表
    # 此辅助函数返回给定state的expected_return
    while True:
        k = np.arange(MAX_CARS + 1)
        states = ((i, j) for i, j in itertools.product(k, k))
        new_values = np.copy(values)  # 用于比对以判断退出迭代
        results = []
        with mp.Pool(processes=MP_PROCESS_NUM) as p:
            f = functools.partial(policy_evaluation_helper,
                                  values=values, policy=policy)
            results = p.map(f, states)
        for v, (i, j) in results:
            new_values[i, j] = v
        diff = np.max(np.abs(values-new_values))
        print('diff in policy_evaluation:{}'.format(diff))
        values = new_values
        if diff <= 1e-1:
            print('Values are converged!')
            return values
```
policy_improvement的实现，将新policy赋为value最大的action。同时返回policy的变化数目，以判断收敛。
```python
def policy_improvement_helper(state, values, actions):
    # 此辅助函数返回给定state的最优action
    # 不够移的action给-inf
    v_max = -float('inf')
    best_action = 0
    for action in actions:
        if ((action >= 0 and state[0] >= action) or (action < 0 and state[1] >= abs(action))) == False:
            v = -float('inf')
        else:
            v = bellman(values, state, action)
        if v >= v_max:
            v_max = v
            best_action = action
    return best_action, state
def policy_improvement(actions, values, policy):
    # 并行的更新policy表 并返回新policy表

    new_policy = np.copy(policy)
   
    results = []
    with mp.Pool(processes=MP_PROCESS_NUM) as p:
        k = np.arange(MAX_CARS + 1)
        states = ((i, j) for i, j in itertools.product(k, k))
        f = functools.partial(policy_improvement_helper,
                              values=values, actions=actions)
        results = p.map(f, states)
    for a, (i, j) in results:
        new_policy[i, j] = a
    policy_change = np.sum(new_policy != policy)
    return new_policy, policy_change
```
最后是solve函数，其中values 和 policy  表都以0作为初值

```python
def solve():
    # 初始化值函数和策略函数表，action表
    values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros_like(values, dtype=np.int)
    actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)  # [-5,-4 ... 4,5]
    iteration_count = 0
    print('Solving...')
    while True:
        start_time = time.time()
        print('#'*10)
        print('Runnning policy_evaluation...')
        values = policy_evaluation(values, policy)
        print('Running policy_improvement...')
        policy, policy_change = policy_improvement(actions, values, policy)
        #print(policy, policy_change)
        #assert False
        iteration_count += 1
        print('iter {} costs {}'.format(iteration_count, time.time()-start_time))
        print('policy_change:{}'.format(policy_change))
        # policy不再变化时终止更新
        if policy_change == 0:
            print('Done!')
            return values, policy
```
最后需如下执行，必需有__main__以防mutiprocessing报错
```python
def main():
    values, policy = solve()
    plot(policy)


if __name__ == '__main__':
    main()
```
此程序在i5-8500执行需要约300s。
