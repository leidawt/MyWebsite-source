---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "python package打包制作"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2020-02-01T12:00:00+08:00
lastmod: 2020-02-01T12:00:00+08:00
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
@[TOC](python package打包制作)
# 1. python package层次结构
根据navdeep-G大神提供的最佳实践（项目模板可从[这里](https://github.com/navdeep-G/setup.py)下载），一个典型python工程项目包应具有如下结构：
{{<figure src = "0.png" title = "" lightbox = "true">}}
其中mypackage是自己要写的包，里面放上__init__.py文件声明该文件夹构成python package，__init__.py可以是个空文件，亦可包含一些import操作，具体取决于我们希望呈现给使用者的使用形式。
当留空__init__.py文件时，我们若想使用a.py 中的a()函数，只需以

```python
from mypackage.a import a
```
方式引入。
而很多时候我们希望能一次引入所有模块，形如我们使用numpy包时先import numpy as np 之后直接按np.xxx() np.random.xxx() 这种方式调用想要的函数，这可通过在__init__.py文件中写入import来实现。如numpy项目在其在顶层__init__.py文件中写入了如下内容：
{{<figure src = "1.png" title = "" lightbox = "true">}}
同时，对应于

```python
from mypackage import *
```
的写法，我们可在__init__.py中写入

```python
__all__ = ['module1', 'module2', 'module1']
```
这个列表，这会指明import *的内容
综上，mypackage的__init__.py文件可如下编写：

```python
__all__ = ['core', 'a', 'b']
from . import core
from . import a
from . import b
```
MANIFEST.in文件是一个清单模板，用于指定要在python源代码分发中分发的其他文件。默认情况下，当实际打包python代码（使用，比方说python setup.py sdist）创建用于分发的打包是，打包程序将仅在包存档中包含一组特定文件（例如，python代码本身）。如果存储库中包含文本文件（例如，模板）或图形（用于您的文档），该怎么办？默认情况下，打包程序不会在归档中包含这些文件，故我们的打包将不够完整，MANIFEST.in 允许覆盖默认值，准确指定打包的文件以供分发。如在上面的模板项目的MANIFEST.in内容为

```python
include README.md LICENSE
```
表示把说明文件README.md和开源协议LICENSE文件一并打包。更多细节可参看[这里](https://docs.python.org/2/distutils/sourcedist.html)。协议文件可参看[这里](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository)来加入。
**更进一步的**，更为规范的项目还会加入单元测试部分和文档部分，文档的自动生成可见[这里](https://pythonguidecn.readthedocs.io/zh/latest/writing/documentation.html#id2)，对使用Sphinx进行文档生成和使用reStructuredText进行文档发布进行了详尽介绍。在构建规范的python包方面，一个十分值得学习的库是[howdoit](https://github.com/gleitz/howdoi/blob/master/howdoi/)，十分规范易懂。此外也可看一下[最佳实践](https://pythonguidecn.readthedocs.io/zh/latest/index.html)。
最后还有一个setup.py文件是打包的关键，下面予以详细讨论。
# 2. python package打包，分发与安装
对于只由单一文件组成的纯python库，我们只需要将python文件拷贝到python安装位置中（如ubuntu一般为/usr/lib/python3.x），即可通过import来找到。
但对于更复杂的由多个文件组成的python库或是含有其他语言编写的库（主要是c,c++），则更推荐使用setuptools提供的工具链打包发布成whl等格式。.whl既是pip管理工具使用的标准库格式，我们可借助pip很方便的部署和管理这个新打包的库，分发也更为容易。
关于setuptools 和 setup.py的详细说明可查阅[cnblogs](https://www.cnblogs.com/cposture/p/9029023.html) 和[this guide](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html)，通常我们只要套用[setup.py](https://github.com/navdeep-G/setup.py)提供的模板即可，如下所示，我们只需要填入meta-data中包名，版本号等信息，并在REQUIRED 中写入依赖的python包即可，这里写入依赖包后，pip install 的时候便会自动检查和安装依赖

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'mypackage'
DESCRIPTION = 'My short description for my project.'
URL = 'https://github.com/me/myproject'
EMAIL = 'me@example.com'
AUTHOR = 'Awesome Soul'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
```
若想推送自己的包到PyPI，则需先注册PyPI账号然后执行python3 setup.py upload来上传。
写好后执行如下命令编译打包为.whl
```bash
python3 setup.py bdist_wheel
```
也可打包为.tar.gz，均可直接由pip工具安装

```bash
python3 setup.py sdist
```

执行setup.py后的目录结构如下：
{{<figure src = "2.png" title = "" lightbox = "true">}}
安装：

```bash
pip3 install ./dist/foo-1.0-py3-none-any.whl #安装包
```

注意，在ubuntu系统中以远程登录方式非root方式执行install时会默认启用--user选项，即会把包安装到用户的~/.local/lib/python3.6/site-packages目录下
{{<figure src = "3.png" title = "" lightbox = "true">}}
而若使用

```bash
sudo pip3 install ./dist/foo-1.0-py3-none-any.whl
```
则会安装到/usr/local/lib/python3.6/dist-packages中去，使用 pip3 show \<packange-name\> 可以看到安装信息



