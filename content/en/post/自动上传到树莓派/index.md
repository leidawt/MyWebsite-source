---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "自动上传到树莓派"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2017-12-06T12:00:00+08:00
lastmod: 2017-12-06T12:00:00+08:00
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
这是一个用python脚本通过sftp快速上传文件夹到树莓派的脚本
可以直接拖拽，会自动上传到树莓派./Desktop/文件夹下
按需修改代码中地址和登录密码即可使用
### 安装依赖
pip instal paramiko
### autoftp.py
```python
#! python3
# -*- coding:utf-8 -*- 
import paramiko
import sys 
import os
import os.path
socks=('192.168.2.100',22)#此处写树莓派的地址
testssh=paramiko.Transport(socks)
testssh.connect(username='pi',password='raspberry')#ssh账号密码
sftptest=paramiko.SFTPClient.from_transport(testssh)
rootdir=sys.argv[1]
rootnamelen=len(rootdir)-len(rootdir.split('\\')[-1])

if os.path.isdir(rootdir) and not rootdir[rootnamelen:] in sftptest.listdir("./Desktop/"):
	tempdir=("./Desktop/"+rootdir[rootnamelen:]).replace('\\','/')
	print("mkdir: ",tempdir)
	sftptest.mkdir(tempdir)
	for parent,dirnames,filenames in os.walk(rootdir):
		for dirname in  dirnames:
			print ("parent is:" + parent+ "  dirname is:" + dirname)
			tempdir=("./Desktop/"+parent[rootnamelen:]+'/'+dirname).replace('\\','/')
			print("mkdir: ",tempdir)
			sftptest.mkdir(tempdir)
		for filename in filenames:
			print ("Upload File: parent is:" + parent +"  filename is:" + filename)
			sftptest.put(parent+'\\'+filename,("./Desktop/"+parent[rootnamelen:]+'/'+filename).replace('\\','/'))
if not os.path.isdir(rootdir):
	sftptest.put(rootdir,"./Desktop/"+rootdir.split('\\')[-1])
	print("upload ",rootdir.split('\\')[-1],"succeed")
else:
	print("dir is already exist")
sftptest.close()
testssh.close()
print("done, sftp closed")

```
### 下面是一个win 下的bat脚本实现拖拽功能
注意将目录换为autoftp.py所在目录
```bat
echo %1
E:
cd E:XXX\XXX
py -3 autoftp.py %1
pause
```
### 然后直接将目标文件夹拖到这个bat上就可以实现自动上传了！
