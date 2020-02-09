---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "stm32 HC05驱动"
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
这个模块提供了格式化发送字符串到HC05的功能，占用stm32的串口3
1.头文件

```c
#ifndef __HC_05
#define __HC_05
/*
本模块为HC05蓝牙透传模块，只写了通信，AT指令部分
单独用串口测试
因此本模块基本就是个串口
占用串口USART3
PB10 TX 5v兼容
PB11 RX 5v兼容


*/
#include "sys.h"
#define HC05_REC_LEN  			100  	//定义最大接收字节数 100
#define HC05_TEC_LEN  			100  	//发送缓存区 100

void initHC05(void);			//初始化
void sendToHC05(char* fmt ,...);//发送到HC05
static void receiveHandler(void);
static void SendCharToHC05(u8 ch);//私有
extern char HC05_RX_BUF[HC05_REC_LEN];
#endif

```
2.c文件

```c
#include "hc05.h"
#include "stdarg.h"
#include "stdio.h"
#include "delay.h"
#include <string.h> 
#include <stdlib.h>
#include "main.h"


//PB10 模块RX
//BB11 模块TX
//注意,读取USARTx->SR能避免莫名其妙的错误   	
char HC05_RX_BUF[HC05_REC_LEN];     //接收缓冲
//接收状态
//bit15，	接收完成标志
//bit14，	接收到0x0d
//bit13~0，	接收到的有效字节数目
u16 HC05_RX_STA=0;       //接收状态标记	  
void initHC05(void){
	GPIO_InitTypeDef  GPIO_InitStructure;
	USART_InitTypeDef  USART_InittStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);
	USART_DeInit(USART3);
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;               
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;      //TX复用推挽输出
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_10MHz;    
	GPIO_Init(GPIOB, &GPIO_InitStructure);               

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;              
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING; //RX浮空输入
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	
	USART_InittStructure.USART_BaudRate = 9600;  //波特率
	USART_InittStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None; //硬件流设置
	USART_InittStructure.USART_Mode = USART_Mode_Rx|USART_Mode_Tx; //接收发送模式
	USART_InittStructure.USART_Parity = USART_Parity_No; //奇偶校验位
	USART_InittStructure.USART_StopBits = USART_StopBits_1; //停止位
	USART_InittStructure.USART_WordLength = USART_WordLength_8b; //字长
	
	USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);  //设置中断类型，接收中断
	
	NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn; //串口3中断，在stm32F10x.h中有定义
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //抢占优先级为3
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;  //响应优先级为3
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //IRQ通道使能

	NVIC_Init(&NVIC_InitStructure);
	USART_Init(USART3, &USART_InittStructure);
	USART_Cmd(USART3, ENABLE);
}
void USART3_IRQHandler(void)
{
	u8 Res;
    if(USART_GetITStatus(USART3, USART_IT_RXNE) != RESET){
		Res =USART_ReceiveData(USART3);	//读取接收到的数据
		if((HC05_RX_STA&0x8000)==0){	//接收未完成
			if(HC05_RX_STA&0x4000){	//接收到了0x0d
				if(Res!=0x0a){
					HC05_RX_STA=0;//接收错误,重新开始
				}
				else {
					//接收完成了
					HC05_RX_STA|=0x8000;
					//完成逻辑:
					printf("recrived: %s",HC05_RX_BUF);
					receiveHandler();
					HC05_RX_STA=0;
					HC05_RX_STA&=~0x8000;//清除完成标志
				}	 
			}
			else {
				//还没收到0X0D	
				if(Res==0x0d){
					HC05_RX_STA|=0x4000;
				}
				else {
					HC05_RX_BUF[HC05_RX_STA&0X3FFF]=Res ;
					HC05_RX_STA++;
					if(HC05_RX_STA>(HC05_REC_LEN-1))HC05_RX_STA=0;//接收数据错误,重新开始接收	  
				}		 
			}
		}
	}
}
//向蓝牙发送一个字节
void SendCharToHC05(u8 ch){      
	while((USART3->SR&0X40)==0)
		;//等待发送完毕   
    USART3->DR = ch;      
}
//格式化发送到蓝牙
void sendToHC05(char* fmt ,...){
	unsigned char i,num;
	char lcd_buf[HC05_TEC_LEN];
	va_list ap;
    va_start(ap,fmt); 
	num=vsprintf(lcd_buf,fmt,ap);
	//printf("num=%d\r\n",num);
	for(i=0;i<num;i++){
		SendCharToHC05(lcd_buf[i]);
		//printf("char=%c\r\n",lcd_buf[i]);
	}
	va_end(ap);
}
//接收回调，处理和解析命令
//指令格式：XXXX;arg0;arg1;arg2
void receiveHandler() {
	//指令表
    const char* list[] = {"setpid", "restart!", "hello","SetCapBeginFlag","restart"};
    unsigned char count = 0;
    double args[3] = {0};
    char* token;
    char* cmd;
    char delim[] = ";";
    int i;
    int length = sizeof(list) / sizeof(char*);
	//先获得命令,';'分割
    cmd = strtok(HC05_RX_BUF, delim);
    printf("cmd= %s\r\n", cmd);
	//然后解析参数，最多3个，解析成double
    token = strtok(NULL, delim);    
    while (token != NULL && count < 3) {
        args[count] = atof(token);
        count++;
        token = strtok(NULL, delim);        
    }
    for (i = 0; i < length; i++) {
        if (strcmp(cmd, list[i]) == 0) break;
    }
	//printf("cmd N=%d\r\n",i);
	//printf("length=%d\r\n",length);
	//按照解析的命令查找执行
    if (i < length) {
        switch (i) {
            case 0:
				pid.Kp = args[0];
				pid.Ki = args[1];
				pid.Kd = args[2];
				printf("setpid:p=%f i=%f d=%f\r\n",args[0],args[1],args[2]);
				savePIDdata();
				__set_FAULTMASK(1);// 关闭所有中断
				NVIC_SystemReset();// 复位
                break;
            case 1:
				__set_FAULTMASK(1);// 关闭所有中断
				NVIC_SystemReset();// 复位
                printf("cmd%d\r\n", i);
                break;
			case 2:
                printf("cmd%d\r\n", i);
				printf("hello world\r\n");
                break;
			case 3:
                printf("cmd%d\r\n", i);
				pauseFlag=0;
                break;
			case 4:
                printf("cmd%d\r\n", i);
				pauseFlag=1;
                break;
            default:
                break;
        }
    }
}

```
IO配置：
PB10 TX 5v兼容
PB11 RX 5v兼容

初始化initHC05()
其中发送可用格式化发送函数sendToHC05，与printf的格式相同
接收需要修改receiveHandler() 函数。接受到数据后会在中断里调用此函数

