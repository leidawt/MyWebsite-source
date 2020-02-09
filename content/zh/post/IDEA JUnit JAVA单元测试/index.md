---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "IDEA JUnit JAVA单元测试"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-02-12T12:00:00+08:00
lastmod: 2018-02-12T12:00:00+08:00
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
#配置
使用插件Junit Generator 来辅助进行测试
##插件配置
设置中搜索 Junit Generator找到配置项，可更改下JUnit 的模板，以解决乱码和依赖，方法为添加import static org.junit.Assert.*;并在下面注释中删除日期，以避免编码问题的乱码	
##使用
选中类名，右键Generate,JUnit4.
在相应位置编写测试：
```java
    /**
     * Method: add2int(int a, int b)
     */
    @Test
    public void testAdd2int() throws Exception {
        assertEquals(3, new MainClass().add2int(1,2));
    }
```
ctrl+shift+F10运行测试
#JUnit
##使用断言
1.  void assertEquals(boolean expected, boolean actual)
检查两个变量或者等式是否平衡
2.	void assertTrue(boolean expected, boolean actual)
检查条件为真
3.	void assertFalse(boolean condition)
检查条件为假
4.	void assertNotNull(Object object)
检查对象不为空
5.	void assertNull(Object object)
检查对象为空
6.	void assertSame(boolean condition)
assertSame() 方法检查两个相关对象是否指向同一个对象
7.	void assertNotSame(boolean condition)
assertNotSame() 方法检查两个相关对象是否不指向同一个对象
8.	void assertArrayEquals(expectedArray, resultArray)
assertArrayEquals() 方法检查两个数组是否相等

##套件测试
测试套件意味着捆绑几个单元测试用例并且一起执行他们。
```java
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
@RunWith(Suite.class)
@Suite.SuiteClasses({
   TestJunit1.class,
   TestJunit2.class
})
public class JunitTestSuite {   
}
```
##其他
@Test(timeout=1000)
来限定执行时间上限
@Test(expected = ArithmeticException.class)
来进行异常测试
#参考
<http://wiki.jikexueyuan.com/project/junit/>
