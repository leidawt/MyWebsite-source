---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "stm32 运行时间测量与间隔执行"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2018-10-15T12:00:00+08:00
lastmod: 2018-10-15T12:00:00+08:00
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
这个程序利用stm32普通定时器构成了执行时间测量功能和间隔执行函数功能，使用如下，两种功能复用，通过#define选择

```c
//执行时间测试模式下 #define StopWatch
//使用：

#include "Runtime.h"

Runtime_init();
while(1){
        Runtime_start();
        delay_ms(1);
        Runtime_stop();
        delay_ms(1000);
}


//setInterval模式 #define setInterval

setInterval(fun,1000);//传入回调，周期1.024*1000ms

```
c文件
```c
#include "Runtime.h"


unsigned int nTime = 0;
void Runtime_init(void) {
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    NVIC_InitTypeDef NVIC_InitStu;
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6, ENABLE);  //使能TIM6时钟
	
	#if defined(StopWatch) 
    /*基础设置，最大可测时间定为8.192ms*/
    TIM_TimeBaseStructure.TIM_Period =
        65536 - 1;  // arr放最大，以实现最大测量范围
	#endif
	
	#if defined(setIntervalMODE) 
    /*1.024ms溢出*/
    TIM_TimeBaseStructure.TIM_Period =
        8192 - 1;  // arr放最大，以实现最大测量范围
	#endif
	
    TIM_TimeBaseStructure.TIM_Prescaler = 9 - 1;                 //预分频
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //向上计数
    TIM_TimeBaseInit(TIM6, &TIM_TimeBaseStructure);

    TIM_ITConfig(TIM6, TIM_IT_Update, ENABLE);  //使能TIM6中断
    TIM_Cmd(TIM6, ENABLE);                      //使能定时器6

    NVIC_InitStu.NVIC_IRQChannel = TIM6_IRQn;  //外部中断线，定时器6
    NVIC_InitStu.NVIC_IRQChannelCmd = ENABLE;
    NVIC_InitStu.NVIC_IRQChannelPreemptionPriority = 1;  //抢占优先级
    NVIC_InitStu.NVIC_IRQChannelSubPriority = 1;         //子优先级
    NVIC_Init(&NVIC_InitStu);
}
#if defined(StopWatch) 
void TIM6_IRQHandler(void) {
    //判断是否为定时器6的更新中断
    if (TIM_GetITStatus(TIM6, TIM_IT_Update) != RESET) {
        nTime++;

        //注意要清除中断标志
        TIM_ClearITPendingBit(TIM6, TIM_IT_Update);
    }
}

void Runtime_start(void) {
    nTime = 0;                //清次数
    TIM_SetCounter(TIM6, 0);  //清空定时器的CNT
}

void Runtime_stop(void) {
    unsigned int count = TIM6->CNT;              // TIM_GetCounter(TIM6);
    TIM_ITConfig(TIM6, TIM_IT_Update, DISABLE);  //关TIM6中断
    printf("run time:%f us %f ms\n", (float)count / 8 + 8192 * nTime,
           (float)count / 8000 + 8.192 * nTime);
    TIM_ITConfig(TIM6, TIM_IT_Update, ENABLE);  //使能TIM6中断
}
#endif

#if defined(setIntervalMODE) 

callbackType callback = NULL;
unsigned int intervalTime=0;

//初始化和设置setInterval
void setInterval(callbackType cb,unsigned int time){
	callback=cb;
	intervalTime=time;
	Runtime_init();
}

void TIM6_IRQHandler(void) {
    //判断是否为定时器6的更新中断
    if (TIM_GetITStatus(TIM6, TIM_IT_Update) != RESET) {
        nTime++;
		if(nTime>=intervalTime){
			TIM_ITConfig(TIM6, TIM_IT_Update, DISABLE);  //关TIM6中断
			nTime=0;
			//需要定时执行的逻辑
			if(callback){
				callback();
			}
			TIM_SetCounter(TIM6, 0);  //清空定时器的CNT
			TIM_ITConfig(TIM6, TIM_IT_Update, ENABLE);  //使能TIM6中断
		}
        //注意要清除中断标志
        TIM_ClearITPendingBit(TIM6, TIM_IT_Update);
    }
}
#endif
```
h文件
```c
#ifndef __RUNTIME__
#define __RUNTIME__

////////MODE SET/////////

//#define StopWatch
#define setIntervalMODE

/////////////////////////

#include "sys.h"
#include "usart.h"


typedef void (*callbackType)(void);

void Runtime_init(void);   //初始化

#if defined(StopWatch)
void Runtime_start(void);  //开始执行时间测试
void Runtime_stop(void);   //结束执行时间测试，打印结果到串口
#endif

#if defined(setIntervalMODE)
void setInterval(callbackType cb,unsigned int time);//初始化和设置setInterval
#endif

#endif

```

