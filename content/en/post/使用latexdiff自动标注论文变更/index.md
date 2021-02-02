---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "使用latexdiff自动标注论文变更"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2021-01-24T12:00:00+08:00
lastmod: 2021-01-24T12:00:00+08:00
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
# 简介
Ref： [Using Latexdiff For Marking Changes To Tex Documents](https://www.overleaf.com/learn/latex/Articles/Using_Latexdiff_For_Marking_Changes_To_Tex_Documents)
latexdiff是CTEX套件中的一款工具，可以自动分析两个tex文件之间的变更情况，生成diff.tex文件，对diff.tex进行编译即可获得可视化的变更标注，如下图所示：
{{<figure src = "0.png" title = "" lightbox = "true">}}
# 使用方法
对修改前tex文件original.tex和修改后tex文件modify.tex进行对比：
命令行（powershell）中执行：
```bash
latexdiff 【original.tex】【modify.tex】> diff.tex
```
成功生成diff.tex文件后，将其编译即可。
# latexdiff+git
Ref: https://tex.stackexchange.com/questions/1325/using-latexdiff-with-git
为实现使用latexdiff工具对比git管理的tex工程的版本，这里编写了一个脚本，定义git ldiff命令，调用latexdiff实现git diff，并用chrome浏览器打开生成的对比文件。
进入git配置文件所在路径，windows为C:\Users\【用户名】
编辑配置文件.gitconfig，在末尾添加：
```bash
# https://tex.stackexchange.com/questions/1325/using-latexdiff-with-git	
[difftool.latex]
	cmd = C:/Users/【用户名】/git-latexdiff.sh \"$LOCAL\" \"$REMOTE\"
[difftool]
	prompt = false
[alias]
	ldiff = difftool -t latex
```
在同一路径放置文件git-latexdiff.sh，内容如下：
```bash
#!/bin/bash
# https://tex.stackexchange.com/questions/1325/using-latexdiff-with-git
# Usage: git ldiff HEAD~1

TMPDIR=$(mktemp -d /tmp/git-latexdiff.XXXXXX)
echo "[INFO] ldiff started"
echo "[INFO] run latexdiff..."
latexdiff "$1" "$2" >$TMPDIR/diff.tex
# sleep 3s
echo "[INFO] run latexdiff done."
echo "[INFO] compiling diff.tex..."
pdflatex -interaction nonstopmode -output-directory $TMPDIR $TMPDIR/diff.tex
echo "[INFO] compiling diff.tex done."
echo "[INFO] opening diff.pdf..."
# evince $TMPDIR/diff.pdf\
if [ -f "$TMPDIR/diff.pdf" ]; then
    chrome.exe --new-window $TMPDIR/diff.pdf
fi
sleep 2s #防止chrome未访问就已经删除pdf文件
echo "[INFO] removing temp files..."
# rm $TMPDIR/*
rm -rf $TMPDIR
```

用法：bash（powershell）中执行：
其中HEAD~x表示同上x个版本对比
```bash
git ldiff HEAD~1
```

