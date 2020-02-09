---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "python笔记"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-09-26T12:00:00+08:00
lastmod: 2018-09-26T12:00:00+08:00
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
@[toc]
### 列表生成器
使用列表生成器可以简便的产生列表，省去for
注意，python3 对其中num变量视作局部变量，而python2则不是，这会影响其上下文的同名变量值！！！
列表生成器和列表推导几乎替代了map reduce filter等高阶函数的绝大多数功能，一般不再使用高阶函数和匿名函数
```python
l = [num*2 for num in range(1, 5)]
print(l)
#[2, 4, 6, 8]
la = [1, 2]
lb = [4, 5, 6]
l = [Point(a, b) for a in la for b in lb]
#[Point(x=1, y=4), Point(x=1, y=5), Point(x=1, y=6), Point(x=2, y=4), Point(x=2, y=5), Point(x=2, y=6)]
```
### 生成器generator
语法与列表生成器类似，仅改为（）。区别在于不一次性计算出列表，二是返回一个可迭代对象generator object，这是可迭代的，可以直接送入max,sum等函数
```python
def a(i):
    return i*2
def b(i):
    return i*3
l = [a, b]
print(list(each(10) for each in l))#[20,30]
print(max(each(10) for each in l))#30
```

### 列表推导
对于过滤等操作很方便
```python
symbols = '$¢£¥€¤'
beyond_ascii = [ord(s) for s in symbols if ord(s) > 127]
#[162, 163, 165, 8364, 164]
beyond_ascii = list(filter(lambda c: c > 127, map(ord, symbols)))
#[162, 163, 165, 8364, 164]
```
### 运算符类与map_reduce
尽管列表推导，生成器代替了绝大多数map_reduce应用，仍有些时候会用到。如计算hash值时需要将一系列值异或起来，如下。
其中operator类提供了所有运算符，避免写lambda函数

```python
import functools
import operator
def hash_all(h):
	hashes = (hash(x) for x in h) # 这里做一个可迭代的生成器
	return functools.reduce(operator.xor, hashes, 0)
	#调用运算符类和reduce算法计算h中所有元素相异或的值
```
### namedtuple命名元组
用于方便的构建仅有几个属性的类，生成的namedtuple可以像类一样引用其属性，如下namedtuple用于表示平面点坐标
```python
import collections
Point = collections.namedtuple('Point', ['x', 'y'])
p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1.x)
```


### 元组

```python
t=(a,b)
first,second=t#拆包
print(t[0])
a, b, *rest = range(5)#平行赋值，多余的存成列表赋给rest
#嵌套的
a=(1,2,(1,2))
q,w,e=a
#q=1 w=2 e=(1,2)
```
### 切片
切片的下标逻辑如下，在索引前切开
{{<figure src = "0.png" title = "" lightbox = "true">}}
还可以用如下的形式对 s 在 a 和 b之间以 c 为间隔取值。c 的值还可以为负
{{<figure src = "1.png" title = "" lightbox = "true">}}
借助切片可方便的赋值
{{<figure src = "2.png" title = "" lightbox = "true">}}
### 排序与插入

```python
fruits = ['grape', 'raspberry', 'apple', 'banana']
#一般排序，原列表并没有变化，返回排好的
print(sorted(fruits))
#传入排序依据函数
print(sorted(fruits, key=len))
fruits.sort()#就地排序


import bisect
import random
random.seed(1729)#输入随机种子
my_list = [random.randrange(1, 10) for each in range(10)]#生成0-9随机类列表
print(my_list)
my_list.sort()#就地排序
bisect.insort(my_list, 3)#按顺序插入（二分）
print(my_list)
```
### 数组
接近c数组，可降低list开销

```python
from array import array
from random import random
floats = array('d', (random() for i in range(100)))#‘d’用于指出存储类型
fp = open('floats.bin', 'wb')
floats.tofile(fp)#可输出为二进制
fp.close()
floats2 = array('d')
fp = open('floats.bin', 'rb')
floats2.fromfile(fp, 100)#可从二进制输入
print(floats[0])#可如列表一样引用
```

### DICT

```python
a = ['a', 'b', 'c']
b = [1, 2, 3]
d = zip(a, b)#返回zip对象，可迭代出(a,b)
dic = dict(d)
print(dic['a'])
dicc=dic


#只读DICT
from types import MappingProxyType
d = {1:'A'}
d_proxy=MappingProxyType(d)
print(d_proxy[1])#只读，不可修改
d[2]='B'
#之后d_proxy会同步更新
```
### SET
set是无重复集。与数学上的集合相似，可进行集合运算

```python
a = [1, 2, 3, 3]
b = [2, 3, 4]
sa = set(a)
sb = set(b)
print(sa | sb)
print(sa-sb)
print(sa & sb)
print(sa ^ sb)#求差集
#另有子集，包含等判断
```
### 函数
这是一个例子，其中
第一个参数后面的任意个参数会被 *content 捕获，存入一个元组
tag 函数签名中没有明确指定名称的关键字参数会被 **attrs 捕
获，存入一个字典。
cls 参数只能作为关键字参数传入。
```python
def tag(name, *content, cls=None, **attrs):
	"""生成一个或多个HTML标签"""
	if cls is not None:
		attrs['class'] = cls
	if attrs:
		attr_str = ''.join(' %s="%s"' % (attr, value)
		for attr, value
		in sorted(attrs.items()))
	else:
		attr_str = ''
	if content:
		return '\n'.join('<%s%s>%s</%s>' %
		(name, attr_str, c, name) for c in content)
	else:
		return '<%s%s />' % (name, attr_str)
```
### 函数内省
在计算机编程中，自省是指这种能力：检查某些事物以确定它是什么、它知道什么以及它能做什么。即当你拿到一个“函数对象”的时候，你可以继续知道，它的名字，参数定义状况等信息。python关于参数信息存放在函数的特殊属性中，时实际不方便使用，借助inspect包可很好处理
内省常用于检查传入的函数是否符合要求

```python
from inspect import signature


def myfun(a, b=1, *c):
    pass


sig = signature(myfun)
print(sig)
for name, param in sig.parameters.items():
    print(param.kind, ':', name, '=', param.default)
```
其中kind有五种
POSITIONAL_OR_KEYWORD
　　可以通过定位参数和关键字参数传入的形参（多数 Python 函数的参
数属于此类）。
VAR_POSITIONAL
　　定位参数元组。
VAR_KEYWORD
　　关键字参数字典。
KEYWORD_ONLY
　　仅限关键字参数（Python 3 新增）。
POSITIONAL_ONLY
　　仅限定位参数；目前，Python 声明函数的句法不支持，但是有些使
用 C 语言实现且不接受关键字参数的函数（如 divmod）支持。
### 函数注解
python3重要的新特性，用于为函数声明中的参数和返回值附加元数据

```python
def clip(text:str, max_len:'int > 0'=80) -> str: ➊
	"""在max_len前面或后面的第一个空格处截断文本
	"""
	end = None
	if len(text) > max_len:
		space_before = text.rfind(' ', 0, max_len)
	if space_before >= 0:
		end = space_before
	else:
		space_after = text.rfind(' ', max_len)
	if space_after >= 0:
		end = space_after
	if end is None: # 没找到空格
		end = len(text)
	return text[:end].rstrip()
```
函数声明中的各个参数可以在 : 之后增加注解表达式。如果参数有默认值，注解放在参数名和 = 号之间。如果想注解返回值，在 ) 和函数声明末尾的 : 之间添加 -> 和一个表达式。那个表达式可以是任何类型。注解中最常用的类型是类（如 str 或 int）和字符串（如 'int >0'）。在示例 5-19 中，max_len 参数的注解用的是字符串。python 对注解所做的唯一的事情是，把它们存储在函数的__annotations__ 属性里
print(signature(f).parameters.values())
可调出属性
### ABC抽象方法
python本身不提供抽象方法，可用ABC类来辅助检查
```python
from abc import ABC, abstractmethod


class Base(ABC):

    @abstractmethod
    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return


class A(Base):
    def load(self, input):
        print(input)


class B(Base):
    pass


a = A()
b = B()#err
```
### 变量作用域&闭包
Python 编译函数的定义体时，默认吧内部变量看做局部变量，若实则指全局，需要显示的声明 global xxx

闭包指延伸了作用域的函数，其中包含函数定义体中引用、但是不在定义体中定义的非全局变量,如

```python
def make_averager():
	series = []
	def averager(new_value):
		series.append(new_value)
		total = sum(series)
		return total/len(series)
	return averager
```
这个函数返回avr函数，其中使用了series存储历史值
当make_averager返回时，它会保留定义函数时存在的自由变量的绑定（此处的series），这样调用函数时，虽然定义作用域不可用了，但是仍能使用那些绑定！

通过使用nonlocal关键字，可不用列表实现上述函数（这样可正确闭包，否则解释器会将count,total视作局部变量，不加以闭包绑定）
```python
def make_averager():
	count = 0
	total = 0
	def averager(new_value):
		nonlocal count, total
		count += 1
		total += new_value
		return total / count
	return averager
```

### 装饰器
装饰器的一大特性是，能把被装饰的函数替换成其他函数。第二个特性是，装饰器在加载模块时立即执行。（函数装饰器在导入模块时立即执行，而被装饰的函数只在明确调用时运行）
**装饰器的典型行为**：把被装饰的函数替换成新函数，二者接受相同的参数，而且（通常）返回被装饰的函数本该返回的值，同时还会做些额外操作。
比如可借助装饰器帮助注册函数到列表
**叠放**：越靠近函数的先起装饰作用
```python
fun_list = []


def isfun(fun):
    fun_list.append(fun)
    return fun


@isfun
def a(i):
    return i*2


@isfun
def b(i):
    return i*3


print(fun_list)
#[<function a at 0x0000000000C2EEA0>, <function b at 0x0000000000C2EF28>]
```
为更好的处理传入函数的参数和特殊属性等，最佳实践是利用functools 中wraps来辅助构建，如下的例子实现了执行时间测试功能，注意传参的处理

```python
import time
from functools import wraps


def clock(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r ' % (elapsed, name, arg_str, result))
        return result
    return clocked


@clock
def fun(a, b, **c):
    print('a=', a, 'b=', b)
    print(c)


fun(1, 2, x=1, y=2)
print(fun.__name__)  # 不改变名称

```
标准库内置了几个很有用的装饰器
**1.**
import functools
@functools.lru_cache()
作用是用缓存优化递归算法（通过保存调用结果），此外lru_cache 在从 Web 中获取信息的应用中也能发挥巨大作用。
可接受参数
functools.lru_cache(maxsize=128, typed=False)
maxsize 参数指定存储多少个调用的结果
typed 参数如果设为 True，把不同参数类型得到的结果分开保存
被 lru_cache 装饰的函数会有 cache_clear 和 cache_info 两个方法，分别用于清除缓存和查看缓存信息。
**2**
@classmethod
用于将方法变成java中静态方法的概念，即不必生成实例即可调用
```python
    @classmethod
    def pargs(self, *args):
        print(*args)
```
### python思维-鸭子类型
“当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来也像鸭子，那么这只鸟就可以被称为鸭子。”
在python的思维中，我们并不关心对象是什么类型，到底是不是鸭子，只关心行为，这一点是和java完全不同的。如下的例子，可见所谓鸭子类型对多态特性的轻巧实现。同样的python提供的大量特殊方法（magic method）也应用了这一思想，如任何实现了 \_\_iter\_\_ 和 \_\_next\_\_方法的对象都可称之为迭代器，而任何类只要实现\_\_getitem\_\_方法，那python的解释器就会把它当做一个collection

```python
class Duck:
    def quack(self):
        print "Quaaaaaack!"
 
class Bird:
    def quack(self):
        print "bird imitate duck."
 
class Doge:
    def quack(self):
        print "doge imitate duck."
 
def in_the_forest(duck):
    duck.quack()
 
duck = Duck()
bird = Bird()
doge = Doge()
for x in [duck, bird, doge]:
    in_the_forest(x)

```

### 对象的一些问题
首先一些概念声明
**==与is**
== 运算符比较两个对象的值（对象中保存的数据），而 is 比较对象的
标识。可以在自己的类中定义 \_\_eq\_\_ 方法，决定 == 如何比较实例。如果不覆盖 \_\_eq\_\_ 方法，那么从 object 继承的方法比较对象的 ID
**浅复制**
如下，只复制浅层，子列表仍指向相同的内存
```python
l1 = [1, 2, [1, 2]]
l2 = list(l1)
print(id(l1) == id(l2))#flase
print(id(l1[-1]) == id(l2[-1]))#true
```
若要深复制，使用copy包的deepcopy函数
**传参**
python传参全部为传引用
这会导致一个不易察觉的问题，即在类中函数用可变对象作为函数的默认参数时，可能会导致多个实例引用到同一个内存上，导致错误。
### class常用特殊方法
特殊方法的使用可使类更符合python的思维，便于接入python的str,sum,==迭代，散列等优秀特性。如下面的类

```python
class Test:
    a = 0
    b = None

    def __init__(self, a, b=None):
        self.a = a
        self.b = b
    # 用于被str()方法调用

    def __str__(self):
        return '__str__ method'
    # 调试信息

    def __repr__(self):
        return '__repr__ method'
    # 用于==运算符

    def __eq__(self, other):
        #print(self.a, other.a)
        return self.a == other.a
    # 定义散列函数，使得可用set

    def __hash__(self):
        return hash(self.a)
    # 迭代。这里返回b的生成器

    def __iter__(self):
        return (each for each in self.b)
    # 使得实例可用len()

    def __len__(self):
        return len(self.b) if self.b is not None else 0
    # 这是声明了静态方法

    @classmethod
    def pargs(self, *args):
        print(*args)

```
另外实现如下的两个特殊方法可实现序列属性，可以实现切片和迭代特性，并可实现[] 访问

```python
class Test:
    _components = []

    def __init__(self, l):
        self._components = l

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        return self._components[index]


t = Test([1, 2, 3])
print(t[0])
for each in t:
    print(each)
```

全部特殊方法：
**流畅的python.pdf-p57**
### 异常
通过捕获异常可以避免脚本直接退出
典型异常处理如下
```python
try:
    fh = open("testfile", "w")
    fh.write("hello world!!")
except IOError:
	#捕获IOError错误
    print("Error: open fail")
else:
	#不发生异常时执行，和try是一对
    print("success!")
    fh.close()
```
python定义好的标准异常有如下这些
http://www.runoob.com/python/python-exceptions.html
可如下发起异常
```python
raise Exception("Invalid level!")
```
或raise其他标准异常，通常不必自定义异常类
### 封包与import
**from…import 与 import**
区别在于import的要用 **模块名.函数名** 来调用，而前者直接用函数名即可。import机制会自动处理重复引用问题，无需多虑
**包**
包就是文件夹，只要该文件夹下存在 \_\_init\_\_.py 文件。
http://www.runoob.com/python/python-modules.html
一般简单的多文件项目不必使用包，直接**import 文件名**即可引入
### 单元测试
使用内置unittest进行
需要import unittest并在测试文件里创建一个测试类。测试文件采用 被测文件名\_test.py命名。测试类继承unittest.TestCase来获得测试方法。测试类的测试函数要以test\_t开头。
有如下常用的测试函数

```python
#测试值相等
self.assertEqual(add(0, 0), 0)
#测试是否正确抛出异常
with self.assertRaises(AttributeError):
    value = d.empty
```

测试例子如下:

```python
#temp.py
def add(a, b):
    return a+b
def sub(a, b):
    return a-b

```


```python
#temp_test.py
import unittest
from temp import add, sub
class TestTest(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(3, 3), 6)
        self.assertEqual(add(4, 3.1), 7.1)
        # self.assertTrue()

    def test_sub(self):
        self.assertEqual(sub(2, 1), 1)

```
在vscode中，会自动识别出测试类，下方信息条展示测试情况
{{<figure src = "3.png" title = "" lightbox = "true">}}
{{<figure src = "4.png" title = "" lightbox = "true">}}
{{<figure src = "5.png" title = "" lightbox = "true">}}
