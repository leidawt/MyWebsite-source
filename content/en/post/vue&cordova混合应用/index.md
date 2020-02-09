---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "vue&cordova混合应用"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-04-01T12:00:00+08:00
lastmod: 2018-04-01T12:00:00+08:00
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
####Vue+VUX ui库+Cordova打包混合应用
####模板工程详见文末pan链接
####1.项目的Cordova基本命令
初始化文件夹
cordova create cordova-app com.lxlazy.www.app cordovaApp 
进入
cd cordova-app 
添加平台使用630API不然找不到
cordova platform add android@6.3.0
检查依赖
cordova requirements
真机调试
cordova run
添加插件
cordova plugin add XXXX
查看安装的插件
cordova plugins
卸载插件
cordova plugin remove XXXX
编译
cordova build android
####2.项目的npm命令
开启开发服务器 npm run dev
编译web文件，更新cordova设置，编译apk并下载到手机运行 npm run update
####3.工程结构
{{<figure src = "0.png" title = "" lightbox = "true">}}
index.html是主入口
src文件夹包含vue组件，vue路由在main.js中，assets存放静态文件。
vue-cordova文件夹存放的是一个vue插件，用于衔接vue和cordova提供的插件。项目地址：[vue-cordova](https://github.com/kartsims/vue-cordova)
####4.添加cordova插件到项目
首先运行cordova plugin add XXXX安装好插件
然后注册到vue-cordova插件方便vue组件使用，如下操作：
{{<figure src = "1.png" title = "" lightbox = "true">}}
以添加蓝牙插件为例：
创建cordova-plugin-bluetooth-serial.js文件到上图位置，内容如下

```
exports.install = function (Vue, options, cb) {
  document.addEventListener('deviceready', () => {

    if (typeof bluetoothSerial === 'undefined') {
      return cb(false)
    }

    // pass through the bluetoothSerial object
    Vue.cordova.bluetoothSerial = bluetoothSerial

    return cb(true)

  }, false)
}
```
然后在上述目录的index.js注册
{{<figure src = "2.png" title = "" lightbox = "true">}}

最后，在vue中使用的方法

```
data: function () {
		return {
			cordova: Vue.cordova
		}
	}
```
插件加载好后会被vue-cordova挂在在这里。
这样拿到后就可以参考cordova的相应文档使用了。
######注意项目中App.vue中mounted内函数对页面的触发控制，只有这样后续才可正常加载！
cordova的插件也可从window上获取，如window.navigator.vibrate(100)
####5.调试
先在dev下完善ui和逻辑，也可在真机上调试web:
打开 Chrome 浏览器，输入地址chrome://inspect，默认进入 chrome://inspect/#devices，将在页面显示当前可用设备，点击链接弹出控制台界面，然后跟普通页面一样调试
之后使用npm run update安装到手机进行混合应用调试
####6.图标
png图标放在res/android目录下，可分大中小，详见cordova
之后修改congig.xml中响应内容即可
####7.一些常用cordova插件
一   震动手机
安装vibrate插件
window.navigator.vibrate(Time in ms);
二  蓝牙串口
安装 bluetoothSerial [项目地址](https://github.com/don/BluetoothSerial)
这是个和蓝牙串口模块通信的插件
使用流程：
获取设备列表    Vue.cordova.bluetoothSerial.list
链接设备   Vue.cordova.bluetoothSerial.connect
向设备发送    Vue.cordova.bluetoothSerial.write
监听设备回复   Vue.cordova.bluetoothSerial.subscribe
三 存储简单数据
简便方法是使用h5的storage特性，无存储期限限制，但大小不得过大，[详细介绍](http://www.w3school.com.cn/html5/html_5_webstorage.asp)
例子如下
```
SaveCurrentData: function () {
	//var value = storage.getItem(key); // 传递键的名字获取对应的值。
	var value = JSON.stringify(this.pineappleNums);//将要存储的内容序列化
	//console.log("SaveCurrentData: ", value);
	localStorage.setItem("pineappleNums", value)
	// 传递键的名字和对应的值去添加或者更新这个键值对。
	//storage.removeItem(key) // 传递键的名字去从LocalStorage里删除这个键值对。
},
restoreData: function () {
	var value = localStorage.getItem("pineappleNums"); // 传递键的名字获取对应的值。
	console.log("restoreData:", JSON.parse(value));
	if (value != null) {
		this.pineappleNums = JSON.parse(value);
	}
}
```
####demo工程下载：
链接：https://pan.baidu.com/s/1wcaCn4PPFGnSxBcYjWFKEg 密码：7w8b
无需再npm install
