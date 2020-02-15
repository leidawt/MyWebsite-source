---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "markdown博客迁移小工具"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-02-11T12:00:00+08:00
lastmod: 2020-02-11T12:00:00+08:00
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
{{% toc %}}

# 动机
鉴于CSDN博客的自动化备份工具基本都挂了，为了简单的备份csdn的markdown博客，编写此小工具，主要用于下载markdown博客里的外链插图到本地，同时做一下链接替换的工作。
原理如下：
1.  使用CSDN编辑器导出博文的markdown
{{<figure src = "0.png" title = "" lightbox = "true">}}
2. 将导出的markdown中的插图下载下来本地化，并进行url替换。
即将原文的
{{<figure src = "1.png" title = "" lightbox = "true">}}
替换为需要的格式，比如hugo博客一般是如下格式：
{{<figure src = "2.png" title = "" lightbox = "true">}}
# 实现
实现很简单，就是正则匹配替换并使用urlretrieve下载图片。核心python脚本如下：
blog_transformer.py
```python
#! py -3
#!/usr/bin/env python3

import re
from urllib.request import urlretrieve
import argparse
import os
import codecs


class BlogTransformer:
    def __init__(self, mode='hugo-academic'):
        """__init__ init class

        Args:
            mode (str, optional): REPLACE mode select. Defaults to 'hugo-academic'.
        """
        self.MD_FILE = ''
        self.SAVE_PATH = ''
        self.REPLACE_MODE = ''
        if mode == 'hugo-academic':
            # https://sourcethemes.com/academic/docs/writing-markdown-latex/#images
            # use format required by hugo
            self.REPLACE_MODE = "{ { { {<figure src = "{0}" title = "" lightbox = "true">} } } }"
        if mode == 'local-img-url':
            # use standard markdown format
            self.REPLACE_MODE = '[]({0})'

    def _pic_download(self, url, path, file_name):
        """_pic_download download picture

        Args:
            url (str): url
            path (str): save path for the picture
            file_name (str): name for the picture

        Returns:
            (save_name, file_type): -
        """
        file_type = ''
        if '.png' in url:
            file_type = '.png'
        elif '.jpg' in url:
            file_type = '.jpg'
        elif '.gif' in url:
            file_type = '.gif'
        else:
            # for some url without expicity file_type, use .png
            file_type = '.png'
        save_name = os.path.join(path, file_name+file_type)
        print('saving to {}'.format(save_name))

        urlretrieve(url, save_name)
        return save_name, file_type

    def run(self, md_file, save_path='', save=True):
        """run download every pic url and modified the markdown

        Args:
            md_file (str): input markdown string
            save_path (str, optional): path to save the modified markdown, when set to default, use the same path as imput markdown file. Defaults to ''.
            save (bool, optional): Save the modified markdown?. Defaults to True.

        Raises:
            ValueError: raise when there are errors in matching pic url, i.e. meet invalid regex match 

        Returns:
            str: modified markdown
        """
        self.MD_FILE = md_file
        if save_path == '':
            # use the same dir as md_file
            self.SAVE_PATH = os.path.dirname(os.path.abspath(self.MD_FILE))
        else:
            self.SAVE_PATH = save_path
        # read markdown file into str
        try:
            with open(self.MD_FILE, 'r') as f:
                md = f.read()
        except:
            # for "utf-8 with dom" format
            with open(self.MD_FILE, 'r', encoding='utf-8-sig') as f:
                md = f.read()
        # findall all ![xxx](xxx) commands
        url_commands = re.findall(r"!\[[\s\S]*?\]\(.+?\)", md)
        # download each img and change the ![xxx](xxx) commands to new REPLACE_MODE format
        for id, each in enumerate(url_commands):
            id = str(id)
            url = re.findall(r"!\[[\s\S]*?\]\((.+?)\)", each)
            if url is not []:
                url = url[0]
                print("Downloading: {}".format(url))

            else:
                raise ValueError('Err in matching url in {}'.format(each))
            save_name, file_type = self._pic_download(
                url, self.SAVE_PATH, id)
            print('Done')
            replace_str = self.REPLACE_MODE.format(id+file_type)
            print('Replacing {} to {}'.format(each, replace_str))
            md = md.replace(each, replace_str)

        # save the new markdown file
        print('#'*80)
        print('Saving modifieded markdown file into {}'.format(
            os.path.join(self.SAVE_PATH, '_'+self.MD_FILE)))
        if save:
            with codecs.open(os.path.join(self.SAVE_PATH, '_'+self.MD_FILE), "w", "utf-8") as f:
                f.write(md)
        print('Done!')
        return md


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  # init
    parser.add_argument('-f', help="input markdown file")
    parser.add_argument('-d', help="path to save pictures", default='')
    parser.add_argument('-m', help="replace mode", default='hugo-academic')
    args = vars(parser.parse_args())

    print('MD_FILE = {}'.format(args['f']))
    print('SAVE_PATH = {}'.format(args['d']))
    print('REPLACE_MODE = {}'.format(args['m']))
    print('#'*80)

    bt = BlogTransformer(args['m'])
    bt.run(args['f'], args['d'])
```
完整代码可见[github](https://github.com/leidawt/blog_transformer)
# 安装使用
## 安装
git clone前述[仓库](https://github.com/leidawt/blog_transformer)，将路径加入环境变量以便bash或命令行能找到。
## 使用
一般，执行下面命令即可，效果是在输入文件YorMarkdown.md的文件夹下导出替换过url的_YorMarkdown.md文件，并下载所有插图到本地，从0开始顺序编号。
```bash
blog_transformer.py -f YorMarkdown.md
```
默认替换为hugo模式的url，亦可使用原生markdown格式：
```bash
blog_transformer.py -f YorMarkdown.md -m local-img-url
```
另一个脚本pipline.py是个更进一步的自动化工具
