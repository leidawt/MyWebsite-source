---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "essential_cpp笔记"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-11-02T12:00:00+08:00
lastmod: 2018-11-02T12:00:00+08:00
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
### 参考与指针
均常用于函数需要传入被修改值时使用，指针与c用法相同而参考（reference)用&表达，如下面的函数用于交换两个值参考指向的是被代指对象，而不再拷贝。

```cpp
void swap(int &val1, int &val2) {
		int temp = val1;
		val1 = val2;
		val2 = temp;
	}
```
相比c风格用指针实现，都达到了传地址的目的，仅写法不同（不必同指针一样用* 和 ->了。
### 堆栈 new delete
这是cpp的动态内存部分，开辟和回收内存。注意配对使用。

```cpp
	int *pi;
	pi = new int;//申请
	delete pi;//释放

	pi = new int[10];//申请
	delete[] pi;//释放
```
### 函数的高级参数写法和应用

```cpp
/*
ex1:给函数添加一个可选的默认关闭的输出流参数
*/
void fun1(ofstream *ofil = 0) {
	if (ofil) {
		(*ofil) << "hello\n";
	}
}
ofstream outfile("data.txt");
fun1(&outfile);//传入
fun1();//默认
```
### 重载函数
同函数名，不同函数参数类型
如

```cpp
void disp(char ch);
void disp(const string&);
```
### 模板函数
提供类型泛型抽象，时常与重载联用

```cpp
//声明一个模板函数
//const强调这里的引用不会改变传入值
template <typename eleType>
void disp(const string &msg, const vector<eleType> &vec) {
	cout << msg <<endl;
	for (unsigned int i = 0; i < vec.size(); i++) {
		eleType temp = vec[i];
		cout << temp <<endl;
	}
}
//测试
	vector <int> a;
	a.push_back(1);
	a.push_back(2);
	a.push_back(3);
	vector <string> b;
	b.push_back("hello");
	b.push_back("world");
	disp("a:",a);
	disp("b:",b);
```
### 函数指针
典型形式如

```cpp
const vector<int>* (*fun)(int);
//这个指针用于指向具有const vector<int>*返回值，有一个int参数的函数
fun=the_fun;//将指针指向the_fun()函数
//应用：
int fun1(int i) {
	return i+1;
}
int fun2(int i) {
	return i+2;
}
//构建以指向fun1类型的函数指针为成员的数组，并初始化
int(*fun_array[])(int) = {
	fun1,fun2
};
//这里enum的使用避免记忆fun1 fun2的数组下标
enum fun_names {
	n_fun1,n_fun2
};
//调用
cout<<fun_array[n_fun1](1);
```
### 迭代器初步

iterator 通常可理解为泛型指针，在构建泛型算法中用处很大。
```cpp
string strs[] = { "hello","world" };
vector<string> vec(strs,strs+2);//利用上面的数组构建向量
vector<string>::iterator iter = vec.begin();//构建一个iterator
while (iter != vec.end()) {
	cout << *iter << endl;
	iter++;
}
```
### 函数对象（function object）
是一个程序设计的对象允许被当作普通函数来调用的特性
c++预置了一些

```cpp
equal_to<type>() 结果为(param1 == param2)
not_equal_to<type>() 结果为(param1 != param2)
less<type>() 结果为 (param1 < param2)
greater<type>() 结果为(param1 > param2)
less_equal<type>() 结果为 (param1 <= param2)
greater_equal<type>() 结果为 (param1 >= param2)
logical_not<type>() 结果为 (!param1)
logical_and<type>() 结果为 (param1 && param2)
logical_or<type>() 结果为 (param1 || param2)
```
可如下使用
```cpp
int a[] = { 7,8,1,2,4,5,10,23,23 };
vector<int> vec(a, a + 9);
//用函数对象做比较函数传入排序算法
sort(vec.begin(), vec.end(), greater<int>());
//遍历打印（use c++11新特性）
for (int &i : vec)
		cout << i << " ";
```
### Function Adapter
用于组合（combine）、变换（transform）、操作（manipulate）函数对象、特定参数值、或者特定函数。
STL提供了一些，如

```cpp
bind2nd(less<int>,val)
//将val 绑定在less的第二个元素上，使得less与val比较
```
### 泛型算法实现样例
以实现泛型的过滤器为例，引入的迭代器起到泛型指针作用，避免对vector int[]等重载函数造成的重复实现，导入的函数指针扩展了过滤逻辑的灵活性

```cpp
//泛型过滤器，接收一对迭代器指向输入
//接收一个比较函数和一个比较值
//接收一个输出迭代器
template<typename Iter,typename ele>
void filter(Iter first, Iter last,Iter out, ele val, bool(*perd)(ele, ele)) {
	while (first != last) {
		if (perd(*first, val)) {
			cout << *first << " ";
			*out++ = *first;
		}
		first++;
	}
	cout << endl;
}
//比较函数
bool myless(int a, int b) {
	return a < b ? true : false;
}
//准备测试数据
const int ele_num = 9;
int a[] = { 7,8,1,2,4,5,10,23,23 };//数据
int a1[ele_num] = { 0 };//接收结果数组
vector<int> vec(a, a + ele_num);//初始化向量
vector<int> vec1(ele_num);//接收向量
filter(vec.begin(),vec.end(),vec1.begin(), 5, myless);//调用
for (int &i : vec1)//遍历打印
	cout << i << " ";
cout << endl;
filter(a, a + ele_num, a1, 10, myless);//调用
for (int &i : a1)//遍历打印
	cout << i << " ";
```
### MAP简介

```cpp
map<string, int> wordlist;
string word;
//这里cin在收到类型不满足string时返回0使得while退出
//通过输入ctrl+z 回车后退出
while (cin >> word)
	wordlist[word]++;//对输入的内容归类统计
for (auto &i : wordlist)//遍历打印，这里i auto为一个pair
	cout << "str= "<<i.first<<" num= "<<i.second << endl;
//wordlist.find("xxx");检索
//wordlist.count("xxx");统计个数
```
### 类
#### 基本
下面的例子展示类基础语法： 构造 解构 成员 方法 静态成员 静态方法

```cpp
//通常下面的类声明放到h文件中
class Test {
public:
	//公开接口和属性
	int a = 0;
	//静态成员，可在所有类实例里共享
	static int static_member;
	//类函数声明
	void print() const;
	//末尾const声明为一const 成员函数，
	//const 关键字只能放在函数声明的尾部，大概是因为其它地方都已经被占用了
	//a. const对象只能访问const成员函数
	//b. const对象的成员是不可修改的,然而const对象通过指针维护的对象却是可以修改的.
	//c. const成员函数不可以修改对象的数据
	//d. 然而加上mutable修饰符的数据成员,对于任何情况下通过任何手段都可修改
	void input_data(int*);
	void print_buf();
	static void print_hello_world();//声明一个静态方法
	//一组构造函数，因为短，习惯上实现和声明都写类里
	//无参数构造函数，以 Test xxx;实例化时使用
	Test() {};
	//一般构造函数
	Test(int num) {
		a = num;
		buf = new int[a];//申请一块内存,长度a
	}
	//一般构造函数
	Test(int num1, int num2) {
		a = num1;
		b = num2;
	}
	//特殊构造函数，处理Test a=b;的初始化方式
	Test(const Test &rhs);
	//解构函数，在类生命周期结束时调用
	~Test() {
		delete[] buf;//释放内存
	}

private:
	//私有
	int b = 0;
	int * buf;
};
//下面这些实现放cpp文件中
//这里的是类方法实现，Test::用于指明其属的类
Test::Test(const Test & rhs) {
	//默认的逐一copy方式不会分配新的heap空间，要手工处理下
	a = rhs.a;
	b = rhs.b;
	buf = new int[a];
	for (int i = 0; i < rhs.a; ++i) {
		buf[i] = rhs.buf[i];
	}
}
void Test::print() const{
	cout << "a=:" << a <<" b=" << b <<endl;
}
void Test::input_data(int *data){
	for (int i = 0; i < a; ++i)
		buf[i] = data[i];
}
void Test::print_buf(){
	for (int i = 0; i < a; ++i)
		cout << buf[i] << " ";
	cout << endl;
}
//静态函数实现时不需要重复添加static关键字
void Test::print_hello_world(){
	cout << "hello world" << endl;
}
//可在此处初始化，不可在函数中进行
int Test::static_member = 1;

//main
int buf[] = { 1,2,3 };
Test t0(3);
t0.print();
t0.input_data(buf);
Test t1 = t0;
t1.print();
t0.print_buf();
t1.print_buf();
Test::print_hello_world();//使用静态函数，类似java static method
cout << Test::static_member << endl;//访问
```

#### 运算符重映射
注：全部的符号重载方法 http://www.runoob.com/cplusplus/cpp-overloading.html

```cpp
class TestOpOrid {
public:
	int num;
	TestOpOrid() {
		num = 0;
	}
	TestOpOrid(int n) {
		num = n;
	}
	//重载加，定义为返回num成员相加的新对象
	TestOpOrid operator+ (const TestOpOrid& rhs) {
		TestOpOrid t;
		t.num = this->num + rhs.num;
		return t;
	}
	//重载==为判断内部num相等
	bool operator== (const TestOpOrid& rhs) {
		return this->num == rhs.num;
	}
};
//mian
TestOpOrid a(1), b(2);
cout << "a==b " << (a == b) << endl;
TestOpOrid c = a + 1;
cout << "c==b " << (c == b) << endl;
return 0;
```
#### 面向对象
c++ 以如下的语法支持面向对象

```cpp

//基础类
class LibMat {
public:
	//定义virtual的方法就可重载了
	virtual void print() const{
		cout << "LibMat print" << endl;
	}
};
//继承LibMat
class Book : public LibMat {
public:
	//注意这里的传参是写作const 引用，否则常数输入会报错
	//另可简单写作Book(const int a,const int b),对传值类型，这样更好
	Book(const int &a,const int &b) {
		this->a = a;
		this->b = b;
	}
	virtual void print() const{
		//可这样调用上层方法
		LibMat::print();
		cout << "Book print" << endl;
	}
	void print_member() {
		cout << a <<','<< b << endl;
	}
protected:
//此关键字表达此2个变量仅可在此派生类使用
	int a, b;
};
//继承机制使得操作一套类相当便利
void display(LibMat & o) {
	o.print();
}

//main
LibMat l;
Book book(1,1);
book.print();
book.print_member();
display(l);
display(book);
```
### 模板类
**tamplete类的声明使用**
因编译的特化原因，通常的做法是将模板类实现直接写在头文件中。
h文件
```cpp
template <typename T>
class BinaryTree {
public:
	BinaryTree();
	~BinaryTree();
	BTnode<T> *_root;
};

template <typename T>
BinaryTree<T>::BinaryTree()
{
}

template <typename T>
BinaryTree<T>::~BinaryTree()
{
}
```

**tamplete 默认参数特性**
{{<figure src = "0.png" title = "" lightbox = "true">}}
调用：
{{<figure src = "1.png" title = "" lightbox = "true">}}
