---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "stm32 12路pwm舵机控制器"
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
用于以pwm信号控制模拟舵机
使用

```c
//@stm32f103 72mhz
PWM_Init(1000-1,1440-1);//开12路pwm 50hz,arr=1000
RATIO_0(150);//设置位置，对模拟舵机CCR:100右极限，200左极限 
```


h文件

```c
#ifndef __pwm_H
#define __pwm_H
#include "sys.h"
/*
使用
PWM_Init(1000-1,1440-1);//开12路pwm 50hz,arr=1000,对模拟舵机
CCR:100右极限，200左极限 RATIO_0(150);//...
*/
#define RATIO_0(RATIO) TIM_SetCompare1(TIM3, RATIO)   // PA6
#define RATIO_1(RATIO) TIM_SetCompare2(TIM3, RATIO)   // PA7
#define RATIO_2(RATIO) TIM_SetCompare3(TIM3, RATIO)   // PB0
#define RATIO_3(RATIO) TIM_SetCompare4(TIM3, RATIO)   // PB1
#define RATIO_4(RATIO) TIM_SetCompare1(TIM4, RATIO)   // PB6
#define RATIO_5(RATIO) TIM_SetCompare2(TIM4, RATIO)   // PB7
#define RATIO_6(RATIO) TIM_SetCompare3(TIM4, RATIO)   // PB8
#define RATIO_7(RATIO) TIM_SetCompare4(TIM4, RATIO)   // PB9
#define RATIO_8(RATIO) TIM_SetCompare1(TIM5, RATIO)   // PA0
#define RATIO_9(RATIO) TIM_SetCompare2(TIM5, RATIO)   // PA1
#define RATIO_10(RATIO) TIM_SetCompare3(TIM5, RATIO)  // PA2
#define RATIO_11(RATIO) TIM_SetCompare4(TIM5, RATIO)  // PA3

void PWM_Init(u16 arr, u16 psc);  //频率为7200000/psc/arr
#endif

// TIM_SetCompare2(TIM3,led0pwmval);调占空比

```
c文件

```c
#include "pwm.h"

//TIM_SetCompare2(TIM3,led0pwmval);调占空比
//PWM频率 = 72M / ((arr+1)*(psc+1))(单位：Hz)
//PWM占空比 = TIM3->CCR1 / arr(单位：%)

//TIM3 TIM4 TIM5 12路 PWM初始化 
//PWM输出初始化
//arr：自动重装值
//psc：时钟预分频数
void PWM_Init(u16 arr,u16 psc)
{  
	GPIO_InitTypeDef GPIO_InitStructure;
	TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	TIM_OCInitTypeDef  TIM_OCInitStructure;
	
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3|RCC_APB1Periph_TIM4|RCC_APB1Periph_TIM5, ENABLE);	//使能定时器345时钟
 	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA|RCC_APB2Periph_GPIOB| RCC_APB2Periph_AFIO, ENABLE);  //使能GPIO外设和AFIO复用功能模块时钟
	
   //设置该引脚为复用输出功能,输出TIM3 CH2的PWM脉冲波形	GPIOA.7
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_2|GPIO_Pin_3|GPIO_Pin_6|GPIO_Pin_7; //A组上管脚
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;  //复用推挽输出
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_10MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);//初始化GPIOA组
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0|GPIO_Pin_1|GPIO_Pin_6|GPIO_Pin_7|GPIO_Pin_8|GPIO_Pin_9;//B组上管脚
	GPIO_Init(GPIOB, &GPIO_InitStructure);//初始化GPIOB组
 
   //初始化TIM3,4,5
	TIM_TimeBaseStructure.TIM_Period = arr; //设置在下一个更新事件装入活动的自动重装载寄存器周期的值
	TIM_TimeBaseStructure.TIM_Prescaler =psc; //设置用来作为TIMx时钟频率除数的预分频值 
	TIM_TimeBaseStructure.TIM_ClockDivision = 0; //设置时钟分割:TDTS = Tck_tim
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //TIM向上计数模式
	TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure); //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
	TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure); //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
	TIM_TimeBaseInit(TIM5, &TIM_TimeBaseStructure); //根据TIM_TimeBaseInitStruct中指定的参数初始化TIMx的时间基数单位
	
	//初始化TIM3,4,5 Channel,2,3,4 PWM模式	 
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1; //选择定时器模式:TIM脉冲宽度调制模式2
 	TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable; //比较输出使能
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High; //输出极性:TIM输出比较极性高
	//根据T指定的参数初始化外设通道，共12个
	TIM_OC1Init(TIM3, &TIM_OCInitStructure);
	TIM_OC2Init(TIM3, &TIM_OCInitStructure);
	TIM_OC3Init(TIM3, &TIM_OCInitStructure);
	TIM_OC4Init(TIM3, &TIM_OCInitStructure);
	TIM_OC1Init(TIM4, &TIM_OCInitStructure);
	TIM_OC2Init(TIM4, &TIM_OCInitStructure);
	TIM_OC3Init(TIM4, &TIM_OCInitStructure);
	TIM_OC4Init(TIM4, &TIM_OCInitStructure);
	TIM_OC1Init(TIM5, &TIM_OCInitStructure);
	TIM_OC2Init(TIM5, &TIM_OCInitStructure);
	TIM_OC3Init(TIM5, &TIM_OCInitStructure);
	TIM_OC4Init(TIM5, &TIM_OCInitStructure);
	//使能TIM 在CCR1,2,3,4上的预装载寄存器
	TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Enable);
	TIM_OC2PreloadConfig(TIM3, TIM_OCPreload_Enable);
	TIM_OC3PreloadConfig(TIM3, TIM_OCPreload_Enable);
	TIM_OC4PreloadConfig(TIM3, TIM_OCPreload_Enable);
	TIM_OC1PreloadConfig(TIM4, TIM_OCPreload_Enable);
	TIM_OC2PreloadConfig(TIM4, TIM_OCPreload_Enable);
	TIM_OC3PreloadConfig(TIM4, TIM_OCPreload_Enable);
	TIM_OC4PreloadConfig(TIM4, TIM_OCPreload_Enable);
	TIM_OC1PreloadConfig(TIM5, TIM_OCPreload_Enable);
	TIM_OC2PreloadConfig(TIM5, TIM_OCPreload_Enable);
	TIM_OC3PreloadConfig(TIM5, TIM_OCPreload_Enable);
	TIM_OC4PreloadConfig(TIM5, TIM_OCPreload_Enable);

	//使能TIM3,4,5
	TIM_Cmd(TIM3, ENABLE); 
	TIM_Cmd(TIM4, ENABLE);
	TIM_Cmd(TIM5, ENABLE);
	

}



```

