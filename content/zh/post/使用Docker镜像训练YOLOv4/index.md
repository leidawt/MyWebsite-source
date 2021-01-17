---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "使用Docker镜像训练YOLOv4"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-12-14T12:00:00+08:00
lastmod: 2020-12-14T12:00:00+08:00
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
最近做横向需要用YOLOv4迁移训练一个模型，但简单搜索居然没有发现特别好用的Docker镜像，故自制了一个darknet版YOLOv4的Docker镜像，本文记录镜像制作及使用镜像训练YOLOv4的方法。

# 镜像制作
Ref: https://github.com/alexeyab/darknet
**注：制作好的镜像已上传至docker hub，可直接拉取使用：**

```bash
docker pull leidawt/darknet-yolov4
```

依赖环境：

 - Ubuntu 18.04 
 - Docker >19.04 
 - nvidia-container-toolkit

为建立镜像，编写如下Dockerfile脚本文件，以nvidia/cuda:10.1-cudnn7-deve为基础镜像开始构建：
```
FROM nvidia/cuda:10.1-cudnn7-devel #拉取cuda基础镜像
COPY ./sources.list /etc/apt/ #更换国内apt源
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections #设置静默安装
# 设置系统字符集以支持中文
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 安装darknet的必要依赖
RUN echo "45.43.38.82 developer.download.nvidia.cn" >> /etc/hosts && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y  build-essential libopencv-dev\
    cmake python3-opencv python3-dev nano
# 建立操作路径
WORKDIR /training
# 安装pip3并更换清华pip源
COPY ./get-pip.py .
RUN python3 get-pip.py && rm get-pip.py &&  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
# 建立操作路径，拷贝darknet程序源文件
WORKDIR /training/darknet
COPY ./darknet-master/ .
# build在建立镜像后通过容器进行操作
# RUN ./build.sh
```
**注：get-pip.py文件和sources.list文件下载后放置到和Dockerfile同一目录：**
链接: https://pan.baidu.com/s/1dyxf-XV14CQwihfBPtv1Vw 提取码: n8xx 


执行镜像构建：
```bash
docker build -t leidawt/darknet-yolov4 .
```
接下来启动一个带有GPU支持的临时容器对YOLOv4代码进行编译：

```bash
docker run --name temp_yolo --gpus all -it leidawt/darknet-yolov4 /bin/bash
```
进入容器后执行：

```bash
cd /training/darknet
./build
```
编译成功后，按ctrl+D退出容器。
commit容器变更到镜像，并删除临时容器:

```bash
docker commit temp_yolo leidawt/darknet-yolov4:v1
docker container rm temp_yolo
```
至此训练镜像构建完毕。
# 准备数据集和配置文件
YOLO的数据集构成结构如下：
{{<figure src = "0.png" title = "" lightbox = "true">}}
每张数据图片对于一个同名的txt标注文件，标注文件格式如下：

```
<object-class> <x_center> <y_center> <width> <height>
```
object-class：所属类号，如 0，1，2，3
x_center：边界框中心x坐标的相对值（x/imageWidth）
y_center：边界框中心y坐标的相对值（y/imageHeight）
width: 边界框宽度的相对值（w/imageWidth）
height: 边界框高度的相对值（h/imageHeight）
```
0 0.5583817829457363 0.791343669250646 0.05644327860159183 0.07525770480212245
0 0.40043604651162784 0.7761627906976746 0.058139534883721186 0.07751937984496159
0 0.44597868217054265 0.8233204134366926 0.03439922480620154 0.029715762273901842
0 0.47862001943634597 0.8276643990929705 0.03401360544217683 0.032798833819241854
0 0.5117984693877551 0.829385325558795 0.03295068027210881 0.030976676384839644
0 0.463283527696793 0.7846412374473598 0.030369290573372194 0.02166342727567212
0 0.4949435131195335 0.7887917071590541 0.03112852283770652 0.02348558471007449
0 0.48104956268221577 0.7448574667962422 0.035228377065111796 0.027939747327502415
0 0.5184797133138971 0.7485017816650469 0.03841715257531586 0.026724975704567555
0 0.4427842565597667 0.7414156138645934 0.04008746355685127 0.027939747327502575
0 0.481277332361516 0.7006195335276968 0.056942419825072935 0.03057175251052797
0 0.5403456025267249 0.7031503077421445 0.05694241982507282 0.025105280207320942
0 0.4244108357628766 0.6958616780045351 0.05375364431486881 0.031988986070618744
0 0.6153506324404762 0.6361607142857142 0.02162388392857141 0.044022817460317436
0 0.619535900297619 0.5853174603174602 0.025809151785714354 0.051463293650793676
0 0.6211635044642857 0.5298239087301586 0.02813430059523808 0.05270337301587307
0 0.3597005208333333 0.6125992063492063 0.019996279761904805 0.04650297619047616
0 0.3598167782738095 0.5639260912698412 0.023949032738095254 0.053323412698412655
0 0.361328125 0.5132378472222221 0.026971726190476216 0.04991319444444445
```
做好数据集后，下一步准备所需的配置文件
- obj.names（类名）:
	形如：
	```
	button
	```
- test.txt（测试集路径索引）:
	形如：
	```
	data/obj/73.jpg 
	data/obj/28.jpg 
	...
	data/obj/32.jpg
	```

- train.txt（训练集路径索引）:
	形如：
	```
	data/obj/34.jpg
	...
	data/obj/39.jpg 
	data/obj/50.jpg
	```

- obj.data（上述三个文件的索引）:
	```
	classes = 1 
		train  = data/train.txt 
		valid  = data/test.txt 
		names = data/obj.names
	```
- yolov4-custom.cfg （配置文件）
从Github[下载](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-custom.cfg)yolov4-custom.cfg文件，按[此处](https://github.com/alexeyab/darknet#how-to-train-to-detect-your-custom-objects)说明进行修改。为了减少手动修改的工作量，写了个小脚本进行自动化：
	util_yolo_cfg_builder.py:
	```python
	# -*- coding: utf-8 -*-
	import argparse
	
	parser = argparse.ArgumentParser()  # 初始化解析器
	parser.add_argument("--print",
	                    action="store_true",
	                    default=False,
	                    help="Print cfg")
	
	parser.add_argument(
	    "--classes", help="Number of classes in training data.")
	# parser.add_argument("--out", help="Path for result output", default="default")
	args = vars(parser.parse_args())  # 从命令行读参数，解析到args
	classes = int(args["classes"])
	
	with open('./yolov4-custom.cfg', 'r') as f:
	    cfg = f.read()
	
	cfg = cfg.replace("max_batches = 500500",
	                  "max_batches = {}".format(classes*2000))
	cfg = cfg.replace("steps=400000,450000", "steps={},{}".format(
	    int(classes*2000*0.8), int(classes*2000*0.9)))
	cfg = cfg.replace("classes=80", "classes={}".format(classes))
	cfg = cfg.replace("filters=255", "filters={}".format((classes + 5)*3))
	
	if args["print"]:
	    print(cfg)
	with open('./yolov4-custom.cfg', 'w') as f:
	    f.write(cfg)
	```
	使用方法：
	
	```bash
	python util_yolo_cfg_builder.py --classes [类数]
	```
准备预训练权重文件yolov4.conv.137
下载：链接: https://pan.baidu.com/s/13ymluUl4wUe8daqVDQQr5g 提取码: g6en 
# 训练
启动镜像：

```bash
docker run --rm --gpus all -p 8888:8888 -v 【数据集和配置文件所在文件夹】:/my_cfgs -it leidawt/darknet-yolov4:v1 /bin/bash
```
其中bind 8888方便通过网页监控训练进度，使用-v映射所需文件，使用--gpus all指定所需使用的GPU。
进入镜像后，执行下述命令把各文件拷贝到正确的路径：
```bash
cd /my_cfgs
cp obj.names /training/darknet/data/ 
cp test.txt /training/darknet/data/ 
cp train.txt /training/darknet/data/ 
cp obj.data /training/darknet/data/ 
cp lgd.cfg /training/darknet/cfg/ 
cp yolov4.conv.137 /training/darknet/ 
mkdir -p /training/darknet/data/obj/ 
mkdir -p /training/darknet/backup 
cp data_dataset_yolo/* /training/darknet/data/obj/
cd -
```
最后，启动训练：
```bash
./darknet detector train data/obj.data cfg/yolov4-custom.cfg.cfg yolov4.conv.137 -dont_show -mjpeg_port 8888 -map
```
通过浏览器访问【宿主机IP:8888】即可监控实时训练进度。
训练参数文件将自动保存在../backup路径下。
# 推理
单张图片推理：

```bash
./darknet detector test data/obj.data cfg/yolov4-custom.cfg backup/<xxxxxx>.weights -dont_show data/obj/1.jpg   
```

该命令生成一个带预测框的predictions.jpg文件。
批量图片推理：

```bash
./darknet detector test data/obj.data cfg/yolov4-custom.cfg backup/<xxxxxx>.weights -ext_output -dont_show -out test_result.json < data/test.txt
```



