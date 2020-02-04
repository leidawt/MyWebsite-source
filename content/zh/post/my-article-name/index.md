---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "样例文章"
subtitle: ""
summary: ""
authors: ["admin"]
tags: ["test"]
categories: []
date: 2020-02-04T17:10:37+08:00
lastmod: 2020-02-04T17:10:37+08:00
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
links:
  - icon_pack: fab
    icon: twitter
    name: Follow
    url: 'https://twitter.com/Twitter'
  - icon_pack: fab
    icon: medium
    name: Originally published on Medium
    url: 'https://medium.com'
---
这是个样板文章，本网站基于Hugo和Academic构建
Hugo 是一个用 Go 编写的静态网站生成器，2013由 Steve Francia 原创，自 v0.14 (2015年) 由 Bjørn Erik Pedersen 主力开发[2]，并由全球各地的开发者和用户提交贡献。Hugo 以 Apache License 2.0 授权的开放源代码项目。[3]

Hugo 一般只需几秒钟就能生成一个网站（每页少于 1 毫秒），被称为“世界上最快的网站构建框架”，也使 Hugo 大受欢迎，成为最热门的静态网站生成器之一，被广泛采用。例如，2015年7月，Netlify 推出专为 Hugo 而设的网站托管服务[4]，而2017年，Smashing Magazine 推出重新设计的官方网站，从原来的 WordPress 迁移到基于 Hugo 的 JAMstack 解决方案。[5]





$$\gamma_{n} = \frac{ 
\left | \left (\mathbf x_{n} - \mathbf x_{n-1} \right )^T 
\left [\nabla F (\mathbf x_{n}) - \nabla F (\mathbf x_{n-1}) \right ] \right |}
{\left \|\nabla F(\mathbf{x}_{n}) - \nabla F(\mathbf{x}_{n-1}) \right \|^2}$$

An example flowchart:

```mermaid
graph TD
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```
An example sequence diagram:

```mermaid
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
    John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```
An example Gantt diagram:
```mermaid
gantt
section Section
Completed :done,    des1, 2014-01-06,2014-01-08
Active        :active,  des2, 2014-01-07, 3d
Parallel 1   :         des3, after des1, 1d
Parallel 2   :         des4, after des1, 1d
Parallel 3   :         des5, after des3, 1d
Parallel 4   :         des6, after des4, 1d
```
An example class diagram:

```mermaid
classDiagram
Class01 <|-- AveryLongClass : Cool
<<interface>> Class01
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData
Class01 : size()
Class01 : int chimp
Class01 : int gorilla
class Class10 {
  <<service>>
  int id
  size()
}
```
An example state diagram:

```mermaid
stateDiagram
[*] --> Still
Still --> [*]
Still --> Moving
Moving --> Still
Moving --> Crash
Crash --> [*]
```

