---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "借助Hugo和Academic主题在github"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-02-04T12:00:00+08:00
lastmod: 2020-02-04T12:00:00+08:00
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
@[TOC]
Hexo+github.io是当前最广为人知的个人网站搭建方法，但Hexo的主题一般只适合于写博客，想构建个包含随笔，论文，代码，教程，博客等多重内容的个人网站并不很方便。我们经常可看到学术大牛们都会有个个人网站介绍自己的论文、团队、简历、博客等等的内容，比如[这个](http://hughandbecky.us/Hugh-CV/)，实现这样的网站使用hexo上的各种主题配合插件自己折腾就稍显麻烦了，因此我们介绍Hugo+Academic主题+github.io构建复合型个人网站的方法。Hugo和Hexo很相似，都是静态网页生成器，Hugo基于go语言编写，速度飞快，配合异常好用的Academic主题，可方便的构建网站。

# 1. 准备
首先准备Hugo环境，可参阅官方文档[Install Hugo](https://gohugo.io/getting-started/installing/)进行安装。对于windows系统，我们只需在[github release](https://github.com/gohugoio/hugo/releases)中下载预编译程序即可。**注意Windows平台的一定要下载带有hugo_extended_xxx** 的版本。下载zip中解压到喜欢的地方，如C:\Program Files\Hugo，然后将路径添加到环境变量即可。在命令行中执行hugo version可检查安装是否正确。
接下来下载Academic。只需在自己github中fork一下模板项目[academic-kickstart](https://github.com/sourcethemes/academic-kickstart#fork-destination-box)，git clone到本地，进入文件夹（我们下面称为网站文件夹），执行
```bash
git submodule update --init --recursive
```
上面语句目的是将仓库所有子模块更新到最新版本。关于git子模块的知识建议查阅[here](https://juejin.im/post/5ca47a84e51d4565372e46e0)。简言之，git子模块是为一个Git仓库中添加其他Git 仓库的场景设计的机制。
上面的仓库用于存放所有网站源码，接下来我们再在github上创建一个托管网站内容文件的仓库。建立一个名字为\<你的github用户名\>.github.io的仓库（注：这是github page要求的形式），然后拉到本地：进入刚才的网站文件夹执行
```bash
git submodule add -f -b master https://github.com/<你的github用户名>/<你的github用户名>.github.io.git public
```
克隆到网站文件夹的public文件夹中。
完成后应得到如下结构
{{<figure src = "0.png" title = "" lightbox = "true">}}
之后，我们将域名填入配置文件：打开 \<网站文件夹\>\config\_default\config.toml，将baseurl配置项写入\<你的github用户名\>.github.io
{{<figure src = "1.png" title = "" lightbox = "true">}}
最后我们进行commit，分两步，一是提交整个网站文件夹用以备份，二是提交到github.io的内容仓库。
在网站文件夹执行以下内容
```bash
git add .
git commit -m "Initial commit"
git push -u origin master

hugo
cd public
git add .
git commit -m "Build website"
git push origin master
cd ..
```
hugo命令会从源码生成静态网站文件到public文件夹。第二个push会要求输入github账号和密码。
至此整个准备和部署已经完成，我们建立好了两个仓库，现在可从 https://<你的github用户名>.github.io 看到初始网页了。
# 2.基本使用及配置
**开启本地测试服务器**
输入hugo server来启动测试服务器，@http://localhost:1313
hugo server会自动侦测源文件变动自动刷新页面，调试十分方便。
**个性化配置**
Academic主题的[官方文档](https://sourcethemes.com/academic/docs/)极为清晰，这里只做下文档导读
我们需要动的内容都集中在confg和content下面，Academic的配置文件采用toml，一个改进了yaml的新的文档格式，并不复杂，配置项的注释里都写明了文档链接，顺次捋一遍按自己需求修改即可。和hexo类似，Academic的内容由markdown文件表达，前面部分用 +++ 包起来的是用于指挥渲染的头信息，后面是正常的markdown内容。home文件夹下是首页各个组件的.md文件，我们可以调整各个组件.md文件中的active配置项来决定是否使用组件。
**开启中英双语言**
幸运的是hugo的多语言支持相当不错。首先修改配置文件，对config\_default\languages.toml修改如下：

```yaml
# Languages
#   Create a `[X]` block for each language you want, where X is the language ID.
#   Refer to https://sourcethemes.com/academic/docs/language/

# Configure the English version of the site.
[en]
  languageCode = "en-us"
  contentDir = "content/en"  # Uncomment for multi-lingual sites, and move English content into `en` sub-folder.

# Uncomment the lines below to configure your website in a second language.
[zh]
  languageCode = "zh-Hans"
  contentDir = "content/zh"
  title = "Academic"
  [zh.params]
    description = "Site description in Chinese..."

  [[zh.menu.main]]
    name = "主页"
    url = "#about"
    weight = 10

  [[zh.menu.main]]
    name = "文章"
    url = "#posts"
    weight = 20

  [[zh.menu.main]]
    name = "项目"
    url = "#projects"
    weight = 30

  [[zh.menu.main]]
    name = "论文"
    url = "#featured"
    weight = 40

  [[zh.menu.main]]
    name = "联系我"
    url = "#contact"
    weight = 60

  [[zh.menu.main]]
    name = "教程"
    url = "courses/"
    weight = 80
```
后面的一大堆是用来汉化主页的bar的，至于为什么Academic本身的汉化包不含这些的原因是因为Academic支持自己在首页定义新的组件，因此bar的内容不定，需要自己汉化。之后把content文件夹的所有内容移动到content\en中，并将content\en文件夹copy一份重命名为content\zh
{{<figure src = "2.png" title = "" lightbox = "true">}}
至此我们就完成了多语言配置，可在网页右上角看到切换选项
{{<figure src = "3.png" title = "" lightbox = "true">}}
开启后两种语言的内容间完全解耦，不存在任何相互关系，需各自单独维护！
Academic为不同内容提供了方便的样式模板，其中最常用的是博客（post）和论文（publication）。
**新建博客（post）**
Academic原生完美支持$\LaTeX$公式渲染，内建的甘特图等图表渲染也十分漂亮，如下：
{{<figure src = "4.png" title = "" lightbox = "true">}}
{{<figure src = "5.png" title = "" lightbox = "true">}}
详情可参考[官方文档](https://sourcethemes.com/academic/docs/writing-markdown-latex/)
新建一篇博客文章可执行：
```bash
hugo new  --kind post post/my-article-name
```
该命令的效果实际会建立\content\en\post\my-article-name文件夹，编辑其中的_index.md即可。hugo new命令并没有考虑多语言，要想建立其中文版，我们需要手工复制一份到\content\zh\post\里去。文章写好后我们只需执行hexo命令进行渲染，之后用git提交到远程库即可。
**新建论文（publication）**
新建一篇博客文章可执行：
```bash
hugo new --kind publication publication/<my-publication>
```
论文类型对论文有专门的排版优化，很适合用于介绍自己的文章。官方文档介绍的python工具可通过bib文件自动生成对应的markdown，但问题很多，当前不建议使用，最好手工编写。论文的显示效果如下：
在首页：
{{<figure src = "6.png" title = "" lightbox = "true">}}
文章：
{{<figure src = "7.png" title = "" lightbox = "true">}}
除了最常用的博客和论文样式，Academic还提供了slide page等丰富的内容，详情可参阅文档。
# 3. 优秀参考范例
https://skyao.io/
https://sourcethemes.com/academic/
https://hughandbecky.us/Hugh-CV/
http://cicl.stanford.edu/
