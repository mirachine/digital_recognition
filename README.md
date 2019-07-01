# 手写数字识别程序(digital_recognition)
不使用深度学习库,手动搭建三层神经网络,进行0到9十分类任务.加入了GUI界面可以在上面手写数字,然后可以对上面的数字进行识别或者训练.也可以选择已经写好的单张图片进行识别或者训练.

## 怎么运行?(Run)
在linux下使用命令:
>`python main.py`

## 需要安装一些必要的库
>### tkinter
>>$`sudo apt-get install python-tk`
>### opencv
>>$`sudo pip install opencv-python`
>### numpy
>>$`sudo pip install numpy`
>### scipy
>>$`sudo pip install scipy`
>### PIL
>>$`sudo pip install pillow`

## 该程序参考的一些资料和教程
>### 程序中的核心算法
>>https://www.bilibili.com/video/av42518802/?p=2 
>### 怎么下载安装库
>>https://www.cnblogs.com/liutongqing/p/6412281.html 
>### 从这两个例子中学习到了怎么从numpy.array对象转换到tkimage.PhotoImage对象
>>https://blog.csdn.net/a1_a1_a/article/details/79981788 
>>https://www.jianshu.com/p/3e80a0b49218 
>### tkinter的学习
>>https://www.bilibili.com/video/av4050443/?p=65 
>>https://www.bilibili.com/video/av16942112/?p=13 
>>http://c.biancheng.net/view/2536.html 
>>https://fishc.com.cn/forum-243-3.html 
>### 原来二级包不能"直接"导入,而需要"直接"导入
>>https://blog.csdn.net/happen_if/article/details/83998708 