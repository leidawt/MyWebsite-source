---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "TagSlam光学定位系统部署与调试"
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
TagSlam是以AprilTag为前端构建的SLAM定位系统，得益于AprilTag技术能提供高精锚点，这一系统具有很高的定位精度。本文记录在ROS中使用TagSlam定位的方法
# USB摄像头使用与矫正
1.  Ubuntu下查看USB摄像头设备
Ref: <http://www.1zlab.com/wiki/python-opencv-tutorial/ubuntu-check-usb-camera-device/>
可通过v4l2-ctl --list-devices命令来确认设备号
2. 摄像头接入ROS
安装usb_cam包

```bash
sudo apt-get install ros-melodic-usb-cam
```

<http://wiki.ros.org/usb_cam#Parameters>
软件包提供了实例launch文件usb_cam-test.launch，可从安装位置拷贝出来修改
3. 校准

```bash
sudo apt install ros-melodic-camera-calibration
```

Ref: <http://wiki.ros.org/camera_calibration>
<http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration>
棋盘格板下载：
<http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration?action=AttachFile&do=view&target=check-108.pdf>
开启摄像头节点后，执行下面命令开启校准程序

```bash
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 image:=/usb_cam/image_raw camera:=/usb_cam
```
参数解释：
--size --square表明所使用了棋盘格板类型
image:=为相机图像消息（需rostopic list找出来对应一下，有时发布时会重映射）
camera:=为相机命名空间（需rostopic list找出来对应一下，有时发布时会重映射）
移动标定板收集不同视角图像后，点击校准计算，再点击commit将校准文件保存到~/.ros/camera_info/下
# Tagslam安装
<https://github.com/berndpfrommer/tagslam_root>
手册
<https://berndpfrommer.github.io/tagslam_web/getting_started/>
注意：tagslam项目使用最新的catkin build编译，而非catkin_make，catkin build是catkin_make下一代，不兼容，必须使用不同的工作空间。
对于X86+ubuntu18.04，只需follow上面官网的安装流程即可，若在树莓派上安装，则需编译gtsam依赖库并修改部分源代码：
**附：修改后的源程序代码**
链接: https://pan.baidu.com/s/1RgEhi9P9uDfm9OhOFm6kNA 提取码: 16fa

首先，由于tagslam系统依赖的gtsam库没有arm二进制包，我们需要手动编译安装它
<https://github.com/borglab/gtsam>
<https://gtsam.org/>
临时开辟一块虚拟内存以避免内存溢出
<https://blog.csdn.net/qq__590980/article/details/102556740>

```bash
sudo mkdir /swap 
cd /swap
sudo fallocate -l 2G swapfile
sudo mkswap swapfile
sudo swapon swapfile
# check:
free -h
```
重启后交换空间就会消失，届时直接删去swapfile文件即可
编译gtsam（约需要4h）：

```bash
git clone https://github.com/borglab/gtsam.git
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j1
make install
```
注意这里指定了编译类型为Release，以提高运行速度
下面开始编译安装tagslam，直接使用修改好的源码src.zip，主要针对boost::irange函数在32位arm平台的错误调用进行了适应性修正

```bash
mkdir xxx/tagslam_root
cd xxx/tagslam_root
mv xxx/src.zip xxx/tagslam_root/
unzip src.zip 
rm src.zip
catkin config -DCMAKE_BUILD_TYPE=Release
catkin build -j1
```
编译约需要8h。
注：若提示找不到Python.h则补充安装如下包即可（注意替换3.5为所使用的python版本）：

```bash
sudo apt-get install python3.5-dev
```
# ROS多工作空间覆盖导致部分工作空间环境变量不生效问题
Ref: https://blog.csdn.net/qq_38441692/article/details/105936291
假设~/catkin_make是主工作空间（其他工作空间由本空间source）
则修改~/catkin_make/devel/_setup_util.py：
{{<figure src = "0.png" title = "" lightbox = "true">}}
更改完后再次编译下工作空间就能实现这个工作空间对其他工作空间进行统一管理了
可以通过

```bash
echo $ROS_PACKAGE_PATH 
```
检查是否同时包含了两个工作空间
# Tagslam配置
基本思想
{{<figure src = "1.png" title = "" lightbox = "true">}}
要使用该系统进行全局定位，首先需要确定相机参数（标定）。其次，在tagslam中所有tag应固定不动，它们相对于lab坐标系（名字其实无所谓）。我们只需指定某个tag相对lab坐标系的位姿即可，当相机同时看到两个以上tag时，其相对位姿会被自动测出并记录下来，非常方便。下面的配置中我指定tag0在lab坐标系远点，tag类型为边长8cm的36h11二维码，使用一只640*480的usb摄像头。
1. 修改apriltag_detector_node.launch文件：
找到工作空间中tagslam项目的launch文件地址（xxx/tagslam_root/src/tagslam/launch/）
编辑apriltag_detector_node.launch文件，将最后的remap按相机节点的消息名称修改,
如：
<remap from="~image/compressed" to="/usb_cam/image_raw/compressed"/>
找到工作空间中tagslam项目的example目录（tagslam_root/src/tagslam/example/），这是官方的demo，写好了一些配置文件，为方便，我们直接将其略加修改即可使用，不必另起炉灶。
2. 编辑cameras.yaml文件：参考
<https://berndpfrommer.github.io/tagslam_web/intrinsic_calibration/>
这个文件的作用是确定相机内参等信息。注意rostopic字段要填写上自己摄像头的消息名（和apriltag_detector_node.launch中的一致）
3. 编辑tagslam.yaml文件：
这个文件描述tagslam系统的主要选项，详见
<https://berndpfrommer.github.io/tagslam_web/input_files/>
其中大部分不需调整，仅需注意一下几个选项：
default_tag_size: 0.08
size: 0.08
这两个参数单位是米，要和使用的apriltag二维码尺寸相对应起来
4. 最后我们准备需要使用的二维码：
<https://berndpfrommer.github.io/tagslam_web/making_tags/>
测试表明，对于640*480分辨率的摄像头，制作（打印）一些8cm的二维码就足以满足的检测需要
注：2，3两步修改过的文件可在文末下载
# Tagslam实验
准备工作进行完毕后，先设一下模式：

```bash
rosparam set use_sim_time true
```
开启摄像头节点

```bash
roslaunch atmosbot_chassis my_cam.launch
```

然后依次开启两个launch

```bash
roslaunch tagslam tagslam.launch run_online:=true
roslaunch tagslam apriltag_detector_node.launch
```

最后启动rviz看一下可视化效果

```bash
rviz -d `rospack find tagslam`/example/tagslam_example.rviz &
```

方便起见，将这些命令写为launch文件保存到
xxx/tagslam_root/src/tagslam/example/example_main.launch

```xml
<launch>
    <include file="$(find atmosbot_chassis)/launch/my_cam.launch">
    </include>
    <include file="$(find tagslam)/launch/tagslam.launch">
        <arg name="run_online" value="true"/>
    </include>
    <include file="$(find tagslam)/launch/apriltag_detector_node.launch">
    </include>
    <node name="tagslam_example_rviz" pkg="rviz" type="rviz" args="-d $(find tagslam)/example/tagslam_example.rviz" required="true" />
</launch>
```
调用时直接运行

```bash
roslaunch tagslam example_main.launch
```

即可。
最后附上配置文件：
链接: https://pan.baidu.com/s/1gDKZ1iPiN5OGHUsnms6kKA 提取码: wev2 

