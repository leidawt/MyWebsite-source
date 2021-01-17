---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "在STM32上运行ROS节点-rosserial-stm32开发及调试方法"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-11-14T12:00:00+08:00
lastmod: 2020-11-14T12:00:00+08:00
featured: false
draft: false
markup: blackfriday

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
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
{{% toc %}}
近期接手了一些ROS机器人项目，这里将开发中遇到的问题和解决方法记录下来。
本文简述借助rosserial在stm32系列中运行ROS节点的方法，stm32强大的外设资源为机器人底层设备控制带来了极大的便利
# 基本原理
Ref: http://wiki.ros.org/rosserial
简言之，rosserial提供一种嵌入式节点和运行在主控PC上的ros master通信的方式，使得在嵌入式节点上编写、运行ros节点成为可能，主要用于转接IO和各类传感器，运行底层控制算法。
要使用rosserial，显然需要分别在嵌入式板卡和主控PC上部署，当前支持的板卡有：
{{<figure src = "0.png" title = "" lightbox = "true">}}
# 安装
由前所述，需要分别在服务端和嵌入式端安装：
## 服务端（PC，树莓派等）安装
```bash
sudo apt-get install ros-melodic-rosserial
sudo apt-get install ros-melodic-rosserial-arduino
```
将库https://github.com/yoneken/rosserial_stm32 clone到catkin_ws/src
这些软件包提供转发功能
## 嵌入式端安装配置（stm32模板工程的建立）
Rosserial在嵌入式端的文件有两部分，一是STM32Hardware.h 和ros.h，这两个文件涉及硬件底层串口的实现，另一部分是ros头文件组，这些文件和在ROS PC端调用的并无区别，官方提供脚本生成这些头文件，使用时只需拷入自己的嵌入式工程目录即可。
因此，关键在于适应性的修改STM32Hardware.h 和ros.h使得rosserial能够调用嵌入式板卡的串口，对此，官方已经提供修改好的文件及样板工程：
https://github.com/yoneken/rosserial_stm32
官方工程基于stm32 HAL库+cubeMX配置
接下来在官方文件的基础上，在最新的STM32CubeIDE1.4.0上建立工程。
**注：关于STM32CubeIDE：近年，ST公司将STM32CubeMX进一步升级，集成到全新的STM32CubeIDE中，STM32CubeIDE是基于eclipse开发的完整且全面的集成环境**

如下是两个关于cubeMX工具和HAL库的极好教程，本文不再赘述：
http://www.mculover666.cn/posts/578764034/
https://www.waveshare.net/study/portal.php?mod=list&catid=40

此外，STM32CubeIDE默认的代码提示的触发条件比较严苛，可参考https://blog.csdn.net/nopear6/article/details/106255311的方法调整STM32CubeIDE的代码补全策略，提高开发效率。

下面开始建立工程：Ref: https://blog.csdn.net/qq_37416258/article/details/84844051
事实上，我们只需要开启并配置串口2即可，官方提供的文件使用串口2实现rosserial通信接口，注意开启串口DMA和中断。
{{<figure src = "1.png" title = "" lightbox = "true">}}
{{<figure src = "2.png" title = "" lightbox = "true">}}
{{<figure src = "3.png" title = "" lightbox = "true">}}
注：上面引用的博客中，开启TIM2是没有必要的，后半部分也有过时的地方，如头文件的修改，详见本文文件夹下建立好的模板工程
串口配置搞定后，我们进一步生成ros头文件并添加到stm32工程中，这些头文件用于提供自定义消息、ros标准消息的支持：
在服务端：
```bash
rosrun rosserial_stm32 make_libraries.py .
```
这将在当前目录生成Inc/文件夹，里面包含需所有头文件，将这些文件拷贝到工程即可，本文模板工程中放置于RosLibs文件夹
**注意，当PC端添加新包后，需再次执行以更新这些头文件**
之后，在工程中建立User文件夹，将mainpp.h，mainpp.cpp拷入，后续自行编写的节点逻辑在此存放。最后，在工程main.c中调用即可。这里同时存在mainpp 和 main 两个“入口”的目的是方便C/C++混合编译。
最后的最后，设置查找路径：
{{<figure src = "4.png" title = "" lightbox = "true">}}
{{<figure src = "5.png" title = "" lightbox = "true">}}
{{<figure src = "6.png" title = "" lightbox = "true">}}
**注：这里截图中使用了绝对路径，后续已修正为相对路径**
**附：模板工程下载**
链接: https://pan.baidu.com/s/1SNI_tKaFUS6zXoIS0GWRCw 提取码: exq5

# ROS节点编写
Ref: https://sudonull.com/post/31955-Rosserial-STM32
官方例程： 
https://github.com/yoneken/rosserial_stm32/tree/master/src/ros_lib/examples
为控制节点发送频率，我们借助cubeMX默认开启的sysTick定时器实现频率控制。
Ref: http://www.mculover666.cn/posts/4283984198/

节点的逻辑在mainpp.c中实现，其中setup()函数放置初始化工作，loop()中编写周期性消息发送指令，如：
```c
void loop(void)
{
    if (nh.connected()){
        if(HAL_GetTick() - tickstart >= 10){
            // 翻转LED，发布chatter消息，周期1000ms
            HAL_GPIO_TogglePin(GPIOG, GPIO_PIN_3);
            str_msg.data = hello;
            chatter.publish(&str_msg);
            tickstart=HAL_GetTick();
        }
    }
    nh.spinOnce();
}
```
# 联合调试
首先在PC端给权限，USB串口插上后，通常挂载在/dev/ttyUSB0，需要线给下777权限：
sudo chmod 777 /dev/ttyUSB0
也可直接添加用户组，一劳永逸：
https://blog.csdn.net/qq_32618327/article/details/106222196
启动服务端桥接节点：
```bash
rosrun rosserial_python serial_node.py
```
# rosserial错误处理
## 开发ros srv时ERROR: service [/topic] responded with an error: service cannot process request: service handler returned None错误的处理方法
https://github.com/ros-drivers/rosserial/pull/414
where you find working ServiceClient.py (https://github.com/ros-drivers/rosserial/releases/tag/0.7.7)
don't forget to update the Arduino ros_lib after step 8
Step 8.1 Delete Arduino/libraries/ros_lib
Step 8.2 Install ros_lib again (https://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup)
Step 8.3 Make Headers agian (https://wiki.ros.org/rosserial_arduino/Tutorials/Adding%20Custom%20Messages)
Step 8.4 Restart Arduino IDE( If it was Open)
Step 9
即：将现有ServiceClient.py替换为更老版本的，当前版本库的ServiceClient.py存在Bug
## tried to publish before configured topic id 125错误的解决
扩大预设缓冲区，原为512
{{<figure src = "7.png" title = "" lightbox = "true">}}
{{<figure src = "8.png" title = "" lightbox = "true">}}
## Printf打印浮点的方法
{{<figure src = "9.png" title = "" lightbox = "true">}}

