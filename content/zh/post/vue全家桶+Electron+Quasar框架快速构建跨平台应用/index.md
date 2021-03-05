---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "vue全家桶+Electron+Quasar框架快速构建跨平台应用"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2021-02-20T12:00:00+08:00
lastmod: 2021-02-20T12:00:00+08:00
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
最近在科研和横向中遇到不少GUI开发的需求，这些项目中，快速、低成本的构建跨平台GUI应用是重中之重，典型的解决方案有Qt和基于web的方案等。鉴于笔者的需求主要是呈现一些科研图表或视频流，并无复杂的交互逻辑，故选择基于web的方案比较合适。具体的，根据之前的使用经验，选择vue来做界面逻辑，vue的生态比较完善，有很多GUI框架可用，并选择Electron打包桌面端。
本文记录学习和使用vue全家桶+Electron+Quasar框架快速构建跨平台应用的一些心得。
# 先导知识
## JS+HTML+CSS三件套
[MDN web docs](https://developer.mozilla.org/zh-CN/docs/Learn/Getting_started_with_the_web) (强烈推荐)
[廖雪峰JS教程](https://www.liaoxuefeng.com/wiki/1022910821149312)
[w3school HTML教程](https://www.w3school.com.cn/html5/index.asp)
[w3school CSS教程](https://www.w3school.com.cn/css3/index.asp)
[阮一峰的网络日志](http://www.ruanyifeng.com/blog/javascript/)
## 前端构建工具
[廖雪峰Node.js教程](https://www.liaoxuefeng.com/wiki/1022910821149312/1023025235359040)
[webpack官方文档](https://www.webpackjs.com/concepts/)
## Electron
[Electron官方文档](https://www.electronjs.org/docs/tutorial/quick-start#%E5%BF%AB%E9%80%9F%E5%90%AF%E5%8A%A8%E6%8C%87%E5%8D%97)
[electron-api-demo](https://github.com/electron/electron-api-demos/releases)：了解electron特性的一个良好选择。
[electron-builder打包见解](https://juejin.cn/post/6844903693683261453)
## vue.js
[vue.js官方文档](https://cn.vuejs.org/v2/guide/)
## UI组件框架
[Vuetify UI库](https://vuetifyjs.com/zh-Hans/introduction/why-vuetify/)：优秀vue组件库，受vue官方推荐
[element UI库](https://element.eleme.cn/#/zh-CN) ：饿了么前端团队推出的组件库
[Quasar UI库](http://www.quasarchs.com/introduction-to-quasar)：十分优秀的多平台UI解决方案
本文使用Quasar框架提供UI支持。Quasar框架是近两年新发展起来的全栈UI框架，组件非常全面，强大，迭代热度很高。
## 常用js库
[axios](https://github.com/axios/axios)：基于promise 的HTTP 库
ffmpeg相关：
1. [node-fluent-ffmpeg](https://github.com/fluent-ffmpeg/node-fluent-ffmpeg)
2. [ffprobe-installer](https://www.npmjs.com/package/@ffprobe-installer/ffprobe)
3. [node-ffmpeg-installer](https://github.com/kribblo/node-ffmpeg-installer)

[lowdb](https://github.com/typicode/lowdb) JSON数据库，适合用于在本地存储小数据
# 环境配置
## 基于vue-cli
安装Node.js环境，[安装包下载](https://nodejs.org/zh-cn/download/)。
为npm包管理器更换华为源（或淘宝源），加速国内访问速度:
```bash
npm config set registry https://mirrors.huaweicloud.com/repository/npm/
npm config set disturl https://mirrors.huaweicloud.com/nodejs/
# electron_mirror 用于加速electron安装
npm config set electron_mirror https://mirrors.huaweicloud.com/electron/
```
全局安装vue-cli工具，该工具为新建vue工程提供了极大的便利。
```bash
npm install -g @vue/cli
```
命令行键入“vue ui”启动vue-cli的可视化界面：
{{<figure src = "0.png" title = "" lightbox = "true">}}
使用默认配置新建工程。
{{<figure src = "1.png" title = "" lightbox = "true">}}
{{<figure src = "2.png" title = "" lightbox = "true">}}
等待CLI工具完成依赖下载和资源配置，随后自动进入项目控制台：
{{<figure src = "3.png" title = "" lightbox = "true">}}
下面通过插件方式安装Quasar UI库，Quasar已经维护了对vue-cli-plugin的支持，非常方便，但官方推荐使用Quasar-cli取而代之。在plugins中搜索并安装vue-cli-plugin-quasar。
{{<figure src = "4.png" title = "" lightbox = "true">}}

添加成功后，编译一下检查配置是否成功。
{{<figure src = "5.png" title = "" lightbox = "true">}}
若一切正常，则可看到demo界面：
{{<figure src = "6.png" title = "" lightbox = "true">}}
用相同的方法添加electron-builder插件。
electron插件会自动配置两个新的任务选项用于编译打包：
{{<figure src = "7.png" title = "" lightbox = "true">}}
首次执行electron:build的时候会从github下载所需的二进制依赖文件，由于众所周知的原因，下载龟速，经常失败。对此可以通过手动下载依赖文件来解决，见[这里](https://www.huaweicloud.com/articles/473ca1dca81f29abc5d721952d3096bc.html)。
所需的依赖文件可从[这里](https://repo.huaweicloud.com/electron/)和[这里](https://mirrors.huaweicloud.com/electron-builder-binaries/)快速下载（ps. 赞一下华为云提供的镜像服务）。
顺利执行打包后，可在【项目路径】/dist_electron下找到打包好的程序。
至此，借助vue-cli工具，我们得以快速完成了程序框架的配置。
## 基于wsl2+quasar-cli（推荐）
### 基础环境配置
quasar框架推荐使用quasar-cli构建应用程序，quasar-cli类似vue-cli，但与quasar框架结合更紧密，省区了很多繁琐的配置。鉴于在windows上安装quasar-cli套件生成的工程的依赖文件时会通过node-gyp编译部分依赖包，存在一些兼容性问题，故笔者转而使用linux环境进行开发，通过WSL2提供ubuntu-18.04环境。请参考[这里](https://docs.microsoft.com/zh-cn/windows/nodejs/setup-on-wsl2)在win10上配置WSL2和node环境。完成node环境配置后，换npm源，安装cli，建立一个demo工程：
```bash
npm config set registry https://registry.npm.taobao.org
npm install -g @quasar/cli
quasar create <folder_name>
```
{{<figure src = "8.png" title = "" lightbox = "true">}}
按提示完成工程配置，最后，quasar-cli会执行npm install安装依赖文件。
启动开发服务器，即可看到demo工程：
```bash
quasar dev
```
{{<figure src = "9.png" title = "" lightbox = "true">}}
### electron环境配置
除了对基础的单页面应用开发的支持，quasar-cli还提供使用electron和Cordova打包PC端和移动端混合应用的能力。如若需添加electron打包功能，执行：
```bash
quasar mode add electron
```
添加electron打包组件，并执行quasar dev -m electron启动开发服务器。
注意，由于WSL2不包含GUI部分，故在打包electron时会遇到一些依赖问题（缺少so包），只需按照报错信息使用apt install对应的包即可。此外，针对GUI程序无法显示的问题，我们可以通过Xserver查看electron程序的GUI界面。使用自带Xserver的客户端访问WSL（如mobaxterm），将如下内容填入~/.bashrc，使得WSL2的图形界面通过Xserver桥接：
```bash
# REF: https://wiki.ubuntu.com/WSL#Running_Graphical_Applications
export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0 # in WSL 2
export LIBGL_ALWAYS_INDIRECT=1
```
之后再执行quasar dev -m electron便可查看electron应用。
{{<figure src = "10.png" title = "" lightbox = "true">}}
若需打包windows平台，则需要额外安装wine组件。笔者按wine官网的标准方式安装，未能成功，后参照[此贴](https://askubuntu.com/questions/1164191/wine-staging-fails-to-install-on-18-04)安装成功，推测安装失败的原因可能与WSL2有关。其步骤如下：
```bash
# 1. 卸载已有安装，删去PPA
sudo apt-get purge *wine*
sudo snap remove wine
sudo snap update wine-platform-*
grep -Ril "wine" /etc/apt
# 2. 准备安装wine
sudo dpkg --add-architecture i386
wget -nc https://dl.winehq.org/wine-builds/winehq.key
sudo apt-key add winehq.key
sudo apt-add-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main'
sudo apt update
sudo apt upgrade
sudo apt --fix-broken install
sudo apt autoremove --purge
sudo apt upgrade
# 3. 安装缺失的依赖
wget https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/Release.key
sudo apt-key add Release.key
sudo apt-add-repository 'deb https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/ ./'
sudo apt update
sudo apt install libfaudio0 libasound2-plugins:i386 -y
# 4. 安装wine
sudo apt install --install-recommends winehq-stable -y
```
### 开发工具配置
推荐使用VScode通过wsl-remote插件直接访问wsl容器，并安装Vetur插件对.vue单文件组件提供支持，安装Vue VSCode Snippets加速开发，安装Eslint提供静态代码分析，安装prettier格式化和工具配合Eslint提供代码格式化。
{{<figure src = "11.png" title = "" lightbox = "true">}}
打开项目文件夹/.vscode/settings.json，添加对项目格式化引擎等选项的配置（仅对本工程生效）：
```bash
{
  "editor.formatOnPaste": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
      "source.fixAll": true
  },
  "javascript.format.insertSpaceBeforeFunctionParenthesis": true,
  "javascript.format.placeOpenBraceOnNewLineForControlBlocks": false,
  "javascript.format.placeOpenBraceOnNewLineForFunctions": false,
  "typescript.format.insertSpaceBeforeFunctionParenthesis": true,
  "typescript.format.placeOpenBraceOnNewLineForControlBlocks": false,
  "typescript.format.placeOpenBraceOnNewLineForFunctions": false,
  "vetur.format.defaultFormatter.html": "prettyhtml",
  "vetur.format.defaultFormatter.js": "prettier-eslint"
}
```
vscode的prettier插件和eslint插件的配置方法见：[here](https://blog.echobind.com/integrating-prettier-eslint-airbnb-style-guide-in-vscode-47f07b5d7d6a)。
### 项目工程结构
{{<figure src = "12.png" title = "" lightbox = "true">}}
其中比较重要的是如下几个目录：
1. /quasar.conf.js 填写quasar框架的配置信息，如，当要使用quasar的对话框插件，需在该文件中添加配置项。
2. /src/router 填写vue router的路由规则
3. /src/layouts 存放描述layouts的.vue文件，该文件描述页面的整体布局（侧边栏，工具栏位置）
4. /src/pages 存放页面，会通过vue路由渲染到/src/layouts/xxx.vue
5. /src/store 存放vuex
6. /src/assets 存放webpack处理的静态资源。这里存放的图片等资源会被wenpack打包处理（如进行base64转换后嵌入js代码），要引用此处存放的资源，应使用
	```html
	<img src="~assets/logo.png">
	```
7. /public 存放静态资源，不被wenpack处理，仅进行复制，故icon等资源一般存放在此处，调用方法如下：
	```html
	<img src="logo.png">
	<!--不要这样使用！！！-->
	<img src="/logo.png">
	```
9. /src-electron 存放electron的配置文件和main thread逻辑
10. /dist/xxx 存放各模式下编译出来的目标文件
# Quasar框架
这里摘录一些Quasar框架常用的功能和组件。
## 实用的css样式类
[字体样式控制](https://quasar.dev/style/typography#Introduction)：提供对文本的字体、字号、加粗、斜体、对齐方式的控制。如，对文本加粗，并居中对齐：
```html
<div class="text-body1 text-weight-bold text-center">Hello Quasar</div>
```
### 颜色系统
[颜色系统](https://quasar.dev/style/color-palette#Introduction)提供了一套主题色和标准色，通过添加text-xxx类，可以变更文字颜色，通过bg-xxx可更换背景色：
```html
<div class="text-body1 text-weight-bold text-center text-primary bg-positive">Hello Quasar</div>
```
{{<figure src = "13.png" title = "" lightbox = "true">}}
[Theme builder](https://quasar.dev/style/theme-builder#Theme-Builder)提供快速配置主题色号的功能。
### Spacing系统
Spacing是指dom元素之间的间隔方式：
{{<figure src = "14.png" title = "" lightbox = "true">}}
[quasar spacing系统](https://quasar.dev/style/spacing#Introduction) 提供一组css类完成spacing工作，具体的像素数会随着响应式系统自动变化，这组类的命名规则如下：
```javascript
q-[p|m][t|r|b|l|a|x|y]-[none|auto|xs|sm|md|lg|xl]
    T       D                   S

T - type
  - values: p (padding), m (margin)

D - direction
  - values:
      t (top), r (right), b (bottom), l (left),
      a (all), x (both left & right), y (both top & bottom)

S - size
  - values:
      none,
      auto (ONLY for specific margins: q-ml-*, q-mr-*, q-mx-*),
      xs (extra small),
      sm (small),
      md (medium),
      lg (large),
      xl (extra large)
# 例如
<div class="q-pa-sm">...</div>
```
### 显示控制
[显示控制css类组](https://quasar.dev/style/visibility#Introduction)用于控制dom元素的显示情况。部分css类的效果如下：
1. disabled 表示禁止选中
{{<figure src = "15.png" title = "" lightbox = "true">}}
2. hidden 表示隐藏该元素
3. invisible 将元素设为不可见，但仍占据原有位置
4. transparent 将组件背景色变为透明（删去背景色）
5. ellipsis ellipsis-2-lines ellipsis-3-lines 将显示不下的长文本省略并在末尾添加省略号
	{{<figure src = "16.png" title = "" lightbox = "true">}}

此外，[显示控制css类组](https://quasar.dev/style/visibility#Introduction)还提供根据window宽度和应用所处的运行平台（手机、pad、桌面端）隐藏和显示组件的功能。
### 实用辅助类
[这些css类](https://quasar.dev/style/other-helper-classes#Introduction)实现了对鼠标选中的控制，滑动的控制，尺寸控制等待。
## 动画
[Animations](https://quasar.dev/options/animations#Introduction)提供了对vue [Transition](https://vuejs.org/v2/guide/transitions.html#Custom-Transition-Classes) 机制和css动效库[Animate.css](https://animate.style/)的封装，使用方法如下：
1. 在/quasar.conf.js中开启Animate.css：
	```javascript
	// 全部载入
	animations: 'all'
	
	// 或选择性载入
	animations: [
	  'bounceInLeft',
	  'bounceOutRight'
	]
	```
2. 使用transition组件包裹需要施加动画的部分
	```html
	<transition
	    enter-active-class="animated fadeIn"
	    leave-active-class="animated fadeOut"
	  >
	    <p v-if="show">hello</p>
	  </transition>
	```
	其中动画选项有六种（实际仅enter-active-class和leave-active-class常用）：
	enter-class
	enter-active-class
	enter-to-class (2.1.8+)
	leave-class
	leave-active-class
	leave-to-class (2.1.8+)
3. 改变组件绑定的show属性，可看到动画效果。
## 布局系统
Quasar的布局系统基于[Flexbox](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)开发，通过Container (Parent)和items (Children)两个层级管理容器中的组件，提供对组件在不同尺寸下排列的方式、位置、间距的精细化控制：
{{<figure src = "17.png" title = "" lightbox = "true">}}
例如，
```html
<div class="row">
  <div class="col-8">two thirds</div>
  <div class="col-2">one sixth</div>
  <div class="col-auto">auto size based on content and available space</div>
  <div class="col">fills remaining available space</div>
</div>
```
其中class="row"是Parent层级，有如下可选类型：
{{<figure src = "18.png" title = "" lightbox = "true">}}
最常用的是row(横向)，column（纵向）。
使用justify类和items-xxx类可控制Parent下子组件横向、纵向的对齐方式：
{{<figure src = "19.png" title = "" lightbox = "true">}}
{{<figure src = "20.png" title = "" lightbox = "true">}}

对与Parent中每个Children组件，“col-xxx” 类用于控制组件的宽（高），Parent被等分为12个块，可使用“col-1”，“col-2”...指定宽高，或使用“col-auto”令Children适应组件内容的形状，“col”则表示指定组件宽为Parent剩余的全部位置。offset-xxx可控制位置的偏置量。
除文档外，Quasar提供[flex-playground](https://quasar.dev/layout/grid/flex-playground#Introduction)帮助用户理解样式的具体效果。
一个例子如下，对组件指定了背景色以更加清楚的显示其占位。
{{<figure src = "21.png" title = "" lightbox = "true">}}

```html
<template>
  <q-page class="q-pa-md column bg-green-2 justify-center items-center">
    <div class="col-auto bg-grey-2">
      <transition
        enter-active-class="animated fadeIn"
        leave-active-class="animated fadeOut"
      >
        <img v-if="show" alt="Quasar logo" src="~assets/quasar-logo-full.svg" />
      </transition>
    </div>

    <div class="col-auto bg-grey-4">
      <q-btn color="secondary" label="Show" @click="show = !show" />
    </div>
  </q-page>
</template>
```
q-page是整个页面的根组件，q-page是页面路由要求使用的。对其添加column使之成为flex布局的parent组件，justify-center items-center使得内部元素垂直居中+水平居中。两个子组件指定col-auto类，以自适应组件内容，否则会有留白，如：将col-auto替换为col（表示填满空间）的效果：
{{<figure src = "22.png" title = "" lightbox = "true">}}
## Layout
QLayout是一个组件，用于管理整个窗口并使用导航栏或抽屉等元素包装页面内容，是对布局系统的一层封装，可以帮助更好地构建网站/应用程序。使用[layout builder](https://quasar.dev/layout-builder)可快速定制一个布局方案。如：
{{<figure src = "23.png" title = "" lightbox = "true">}}
builder生成的代码应填写到/src/layouts/MainLayout.vue路径。对其中的header footer和侧边栏属性的详细控制可在文档中找到。
## 组件库
Quasar提供了海量的vue组件（[组件是可复用的 Vue 实例，详见vue文档](https://cn.vuejs.org/v2/guide/components.html)），文档中，每个组件有四类API：
{{<figure src = "24.png" title = "" lightbox = "true">}}

probs是传入组件的属性，例如向QAjaxBar组件传入skip-hijack等属性，可写作：
```bash
<q-ajax-bar
      position="bottom"
      color="accent"
      size="10px"
      skip-hijack
/>
```
events是组件能够发出的事件。组件事件用于如下场景（[摘自vue文档](https://cn.vuejs.org/v2/guide/components.html#%E7%9B%91%E5%90%AC%E5%AD%90%E7%BB%84%E4%BB%B6%E4%BA%8B%E4%BB%B6)）：
{{<figure src = "25.png" title = "" lightbox = "true">}}
slots用于指定组件分发内容（[文档](https://cn.vuejs.org/v2/guide/components.html#%E9%80%9A%E8%BF%87%E6%8F%92%E6%A7%BD%E5%88%86%E5%8F%91%E5%86%85%E5%AE%B9)）。注意[具名插槽](https://cn.vuejs.org/v2/guide/components-slots.html#%E5%85%B7%E5%90%8D%E6%8F%92%E6%A7%BD)的使用，具名插槽机制可令组件接收多个输入插槽。在向具名插槽提供内容的时候，我们可以在一个 template元素上使用 v-slot 指令，并以 v-slot 的参数的形式提供其名称：
```html
<base-layout>
  <template v-slot:header>
    <h1>Here might be a page title</h1>
  </template>

  <p>A paragraph for the main content.</p>
  <p>And another one.</p>

  <template v-slot:footer>
    <p>Here's some contact info</p>
  </template>
</base-layout>
```
methods是组件提供的分发，通过设置组件的ref属性，我们可通过this.$refs对组件进行索引，进而调用组件方法（[vue文档](https://cn.vuejs.org/v2/guide/components-edge-cases.html#%E8%AE%BF%E9%97%AE%E5%AD%90%E7%BB%84%E4%BB%B6%E5%AE%9E%E4%BE%8B%E6%88%96%E5%AD%90%E5%85%83%E7%B4%A0)）：
如：
```html
<base-input ref="usernameInput"></base-input>
```
```javascript
// 引用
this.$refs.usernameInput.xxx()
```
## 插件
某些功能需要通过[插件](https://quasar.dev/quasar-plugins/)的方式实现，如对话框、通知、全屏触发等常用功能。
## utils
[utils](https://quasar.dev/quasar-utils/)中提供了一些实用的函数，如格式化显示日期、时间，下载触发、url跳转、复制到剪贴板、api触发频率限制、uid生成、对象深拷贝等。
## 页面路由
页面路由是单页应用的重要环节，quasar项目集成vue-router支持。在src/layouts/MainLayout.vue中使用q-route-tab编写路由接口：
```html
<q-tabs v-model="tab" align="left">
  <q-route-tab to="/" label="index" />
  <q-route-tab to="/testpage" label="testpage" />
</q-tabs>
```
{{<figure src = "26.png" title = "" lightbox = "true">}}
在src/router/routes.js中填写路由映射：
```javascript
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      { path: '', component: () => import('pages/Index.vue') },
      { path: 'testpage', component: () => import('pages/TestPage.vue') },
    ],
  },
```
在/src/pages目录下放置页面：
```html
<template>
  <q-page class="flex flex-center">
    <h1>{{ foo }}</h1>
  </q-page>
</template>

<script>
export default {
  name: 'PageTestPage',
  data() {
    return {
      foo: 'this is testpage',
    };
  },
};
</script>
```
## vuex
quaser项目也提供了对vuex的支持（可选），Vuex 是一个专为 Vue.js 应用程序开发的状态管理模式，用于解决多组件之间的共享状态问题。在单页app中，vuex典型的应用场景是同步多个页面之间的状态。在vuex的设计中，通过mutation（一组函数）来变更状态（state），配合vue的计算属性机制，多个页面的组件之间可响应式的共享state的值。{{<figure src = "27.png" title = "" lightbox = "true">}}
quaser中，使用[vuex-模块](https://vuex.vuejs.org/zh/guide/modules.html)进行组织，更易于维护。{{<figure src = "28.png" title = "" lightbox = "true">}}
其中每个模块文件夹包含如下文件：
{{<figure src = "29.png" title = "" lightbox = "true">}}
使用cli可以代替手动复制，快速新建一个模块：
```bash
quasar new store <store_name>
```
下面举例说明如何使用vuex在两个页面之间同步状态。
命令行执行quasar new store showcase生成一个新的showcase模块，编辑src/store/index.js引入新模块：
{{<figure src = "30.png" title = "" lightbox = "true">}}
编辑src/store/showcase/state.js定义两个欲进行同步的状态drawerState、message：
```javascript
export default function () {
  return {
    drawerState: true,
    message: '',
  };
}
```
编辑src/store/showcase/mutations.js编写修改状态的方法：
```javascript
export const updateDrawerState = (state, opened) => {
  state.drawerState = opened;
};
export const updateMessageState = (state, msg) => {
  state.message = msg;
};

```
在页面组件中使用：
页面1：index.vue
```html
<template>
  <q-page class="q-pa-md column justify-center items-center q-gutter-md">
    <div class="col-auto">
      <transition
        enter-active-class="animated fadeIn"
        leave-active-class="animated fadeOut"
      >
        <img v-if="show" alt="Quasar logo" src="~assets/quasar-logo-full.svg" />
      </transition>
    </div>

    <div class="col-auto">
      <q-btn color="secondary" label="Show" @click="show = !show" />
    </div>
    <div class="col-auto">
      {{ drawerState }}
      <q-toggle v-model="drawerState" />
      <q-btn color="secondary" label="Update" @click="updateMsg" />
    </div>
  </q-page>
</template>

<script>
import { date } from 'quasar';

export default {
  name: 'PageIndex',
  data() {
    return {
      show: true,
    };
  },
  methods: {
    updateMsg() {
      const { addToDate } = date;

      const newDate = addToDate(new Date(), { days: 7, month: 1 });
      this.$store.commit('showcase/updateMessageState', newDate);
    },
  },
  computed: {
    drawerState: {
      get() {
        return this.$store.state.showcase.drawerState;
      },
      set(val) {
        this.$store.commit('showcase/updateDrawerState', val);
      },
    },
  },
};
</script>
```
页面2：testpage.vue：
```html
<template>
  <q-page class="flex flex-center">
    <!-- <h1>{{ foo }}</h1> -->
    <q-toggle v-model="drawerState" />
    <p>{{ msgState }}</p>
  </q-page>
</template>

<script>
export default {
  name: 'PageTestPage',
  data() {
    return {
      foo: 'this is testpage',
    };
  },
  computed: {
    drawerState: {
      get() {
        return this.$store.state.showcase.drawerState;
      },
      set(val) {
        this.$store.commit('showcase/updateDrawerState', val);
      },
    },
    msgState() {
      return this.$store.state.showcase.message;
    },
  },
};
</script>

```
通过定义计算属性的get，可以在this.$store.state被变更时自动刷新dom渲染。对状态的变更则一律通过this.\$store.commit调用src/store/showcase/mutations.js中编写的修改状态的方法来实现。效果如下，两个页面的开关状态同步，页面1update变更状态可以在页面2体现：
{{<figure src = "31.png" title = "" lightbox = "true">}}
{{<figure src = "32.png" title = "" lightbox = "true">}}
## electron nodejs api调用
对于添加了electron模块的quasar项目，quasar生成项目时已经进行了相关配置，故可在前端代码直接使用nodejs的模块。
## api调用
Axios是一个基于promise 的著名HTTP库，如在生成工程时选择添加axios，则会在src/boot/axios.js中将axios库挂载到Vue.prototype.\$axios，.vue文件中使用 this.\$axios可访问它。典型使用如下：
```javascript
 this.$axios
    .get('https://api.coindesk.com/v1/bpi/currentprice.json')
    .then(response => {
      this.info = response.data.bpi
    })
    .catch(error => {
      console.log(error)
      this.errored = true
    })
    .finally(() => this.loading = false)
```
注：例子中的coindesk api是一个支持跨域访问的比特币行情api。
注意，不同于jsonp，axios是无法单方面解决跨域问题的，需要接口配合。如，使用python flask写个api接口，可通过cross_origin库添加跨域许可：
```python
from flask import Flask, request
from flask_cors import cross_origin

app = Flask(__name__)


@app.route('/hello')
@cross_origin(resources=r'/*')
def hello():
    return "hello"

if __name__ == '__main__':
    app.run(host='192.168.3.16',port=5000,debug=True)
```
此处服务端使用本机真实IP地址，若使用127.0.0.1，则在electron打包的app中，axios报错（xhr），从axios 的github讨论来看，可能是一个bug。关于IP、localhost、127.0.0.1的区别可参考[此文](https://blog.csdn.net/mengzuchao/article/details/81462958)。
## echart图表
echart是百度开发的前端图表库，已经捐助到apache基金会。
直接使用原生的Echart库，参考文章[vue中使用ECharts实现折线图和饼图](https://segmentfault.com/a/1190000022096665)。
{{<figure src = "33.png" title = "" lightbox = "true">}}
src/pages/echartDemo.vue：
```html
<template>
  <q-page class="flex flex-center">
    <div ref="chartPie" class="pie-wrap"></div>
  </q-page>
</template>

<script>
import * as echarts from 'echarts';

export default {
  name: 'echartDemo',
  data() {
    return {
      chartPie: null,
    };
  },
  mounted() {
    this.$nextTick(() => {
      this.drawPieChart();
    });
  },
  methods: {
    drawPieChart() {
      const mytextStyle = {
        color: '#333',
        fontSize: 18,
      };
      const mylabel = {
        show: true,
        position: 'right',
        offset: [30, 40],
        formatter: '{b} : {c} ({d}%)',
        textStyle: mytextStyle,
      };
      this.chartPie = echarts.init(this.$refs.chartPie);
      this.chartPie.setOption({
        title: {
          text: 'Pie Chart',
          subtext: '纯属虚构',
          x: 'center',
        },
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b} : {c} ({d}%)',
        },
        legend: {
          data: ['直接访问', '邮件营销', '联盟广告', '视频广告', '搜索引擎'],
          left: 'center',
          top: 'bottom',
          orient: 'horizontal',
        },
        series: [
          {
            name: '访问来源',
            type: 'pie',
            radius: ['50%', '70%'],
            center: ['50%', '50%'],
            data: [
              { value: 335, name: '直接访问' },
              { value: 310, name: '邮件营销' },
              { value: 234, name: '联盟广告' },
              { value: 135, name: '视频广告' },
              { value: 1548, name: '搜索引擎' },
            ],
            animationEasing: 'cubicInOut',
            animationDuration: 2600,
            label: {
              emphasis: mylabel,
            },
          },
        ],
      });
    },
  },
};
</script>

<style>
.pie-wrap {
  width: 100%;
  height: 400px;
}
</style>
```
其中， 触发绘制的函数drawPieChart被挂载到mounted()，根据vue生命周期钩子的描述，mounted()将在dom渲染完毕后执行。使用this.\$nextTick方法可以使得dom刷新后drawPieChart再被触发。

## 组件化开发
对于重复的逻辑，可以抽离并编写成自定义组件，提升代码的可维护性。举例来说，构建一个折线图组件，接收x轴图例数组和y轴数据两个属性，渲染一个折线图。
工程的组件放在src/components目录。编辑src/components/EchartsCategory.vue，编写组件：
```html
<template>
  <div ref="chart" class="chart"></div>
</template>

<script>
import * as echarts from 'echarts';

export default {
  name: 'EchartsCategory',
  components: {},
  data() {
    return {
      // echart对象（dom）
      myChart: null,
    };
  },
  props: {
    xAxisdata: {
      type: Array,
      default() {
        return ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
      },
    },
    yAxisdata: {
      type: Array,
      default() {
        return [150, 230, 224, 218, 135, 147, 260];
      },
    },
  },
  mounted() {
    // dom刷新后触发绘制
    this.$nextTick(() => {
      this.draw();
    });
    // this.draw();
  },

  methods: {
    draw() {
      this.myChart = echarts.init(this.$refs.chart);
      this.myChart.setOption({
        xAxis: {
          type: 'category',
          data: this.xAxisdata,
        },
        yAxis: {
          type: 'value',
        },
        series: [
          {
            data: this.yAxisdata,
            type: 'line',
          },
        ],
      });
    },
  },
};
</script>
<style>
.chart {
  width: 100%;
  height: 400px;
}
</style>

```
其中props定义了组件可接收的属性的规格和默认值。
在src/pages/echartDemo.vue中使用EchartsCategory这一组件：
```html
<template>
  <q-page class="flex flex-center">
    <EchartsCategory v-bind="EchartsCategoryData"></EchartsCategory>
  </q-page>
</template>
<script>
import EchartsCategory from 'components/EchartsCategory.vue';

export default {
  name: 'echartDemo',
  components: { EchartsCategory },
  data() {
    return {
      // bind到EchartsCategory组件的模拟数据
      EchartsCategoryData: {
        xAxisdata: ['abc', 'def', 'sdsd'],
        yAxisdata: [12, 35, 76],
      },
    };
  },
};
</script>
```
使用import引入组件后，需在components中加以引用。对EchartsCategory 组件添加v-bind，将props整体传入，这和如下写法是等效的：
```html
<EchartsCategory 
	v-bind:xAxisdata="EchartsCategoryData.xAxisdata"
	v-bind:yAxisdata="EchartsCategoryData.yAxisdata">
</EchartsCategory>
```
