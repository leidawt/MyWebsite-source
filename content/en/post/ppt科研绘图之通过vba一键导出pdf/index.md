---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "ppt科研绘图之通过vba一键导出pdf"
subtitle: ""
summary: ""
authors: ["admin"]
tags: []
categories: []
date: 2021-03-03T12:00:00+08:00
lastmod: 2021-03-03T12:00:00+08:00
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
ppt是画科研插图的优秀工具，能够很方便的导出为pdf并插入latex，但手动通过“另存为-格式-pdf-当前页-确定”将一页ppt保存为一个pdf文件要选择多个选项，稍显繁琐，故编写了一小段vba脚本来自动化这一步骤。
执行该宏，可很方便的将ppt页导出为无边框的pdf文件：
{{<figure src = "0.png" title = "" lightbox = "true">}}
示例文件下载：
链接: https://pan.baidu.com/s/1QGTC5a5kD35lD-7_EdPqpQ 提取码: kd45

首先确保将ppt保存为启用宏的pptm文件格式。{{<figure src = "1.png" title = "" lightbox = "true">}}
开启“开发工具”选项卡
{{<figure src = "2.png" title = "" lightbox = "true">}}
打开vb编辑器填入如下代码并保存：
{{<figure src = "3.png" title = "" lightbox = "true">}}
{{<figure src = "4.png" title = "" lightbox = "true">}}

```vbnet
Sub 逐张存储()
    'https://stackoverflow.com/questions/17929124/export-each-slide-of-powerpoint-to-a-separate-pdf-file
    Dim strNotes As String, savePath As String
    Dim oPPT As Presentation, oSlide As Slide
    Dim sPath As String, sExt As String
    
    Set oPPT = ActivePresentation
    'sPath = oPPT.FullName & "_Slide_"
    sPath = oPPT.Path
    sExt = ".pdf"
    
    For Each oSlide In oPPT.Slides
        i = oSlide.SlideNumber
        oSlide.Select
        strNotes = oSlide.NotesPage. _
            Shapes.Placeholders(2).TextFrame.TextRange.Text
        'MsgBox strNotes
        'MsgBox Len(strNotes) = 0
        savePath = sPath & "\" & strNotes & sExt
        If Not Len(strNotes) = 0 Then
            oPPT.ExportAsFixedFormat _
                Path:=savePath, _
                FixedFormatType:=ppFixedFormatTypePDF, _
                RangeType:=ppPrintSelection
            Shell "pdfcrop " & savePath & " " & savePath
        End If
    Next
    Set oPPT = Nothing
End Sub
Sub 选中存储()
    Dim strNotes As String, savePath As String
    Dim oPPT As Presentation, oSlide As Slide
    Dim sPath As String, sExt As String
    
    Set oPPT = ActivePresentation
    'sPath = oPPT.FullName & "_Slide_"
    sPath = oPPT.Path
    sExt = ".pdf"

    Set oSlide = Application.ActiveWindow.View.Slide
    strNotes = oSlide.NotesPage. _
        Shapes.Placeholders(2).TextFrame.TextRange.Text
    savePath = sPath & "\" & strNotes & sExt
    If Not Len(strNotes) = 0 Then
        oPPT.ExportAsFixedFormat _
            Path:=savePath, _
            FixedFormatType:=ppFixedFormatTypePDF, _
            RangeType:=ppPrintSelection
        Shell "pdfcrop " & savePath & " " & savePath
    End If
    Set oPPT = Nothing
End Sub

```

在宏管理器中双击脚本即可执行。
{{<figure src = "5.png" title = "" lightbox = "true">}}
其中”逐张存储“的功能是遍历ppt中的所有页，逐个保存为pdf文件。其中文件名为每页的备注内容。
{{<figure src = "6.png" title = "" lightbox = "true">}}
如备注为空则跳过该页。此外，对每个保存的pdf文件，还将执行pdfcrop（latex套件提供的一个pdf剪裁工具），将pdf的白边减裁掉，方便贴到latex中。
”选中存储“功能类似，但仅对选中的那一页ppt进行处理。
为更加方便的使用该宏，可将宏添加到自定义菜单。
{{<figure src = "7.png" title = "" lightbox = "true">}}
{{<figure src = "8.png" title = "" lightbox = "true">}}
{{<figure src = "9.png" title = "" lightbox = "true">}}

