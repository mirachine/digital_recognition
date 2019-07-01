'''
Python 3.6.7
UTF-8
GPL-3.0
'''

# gui库
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
# opencv只用来画图了,没用到图像处理的功能
import cv2
# 矩阵运算用到的库
import numpy as np
# 科学计算用到的库,其中激活函数Sigmoid函数在里面
import scipy.special
# 读取图片和转换图片格式用到的库
from PIL import Image, ImageTk

class neuralNetwork :

    # 用于神经网络初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入层节点数
        self.inodes = inputnodes
        # 隐层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 初始化输入层与隐层之间的权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐层与输出层之间的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数（S函数）
        self.activation_function = lambda x: scipy.special.expit(x)

    # 设置权重
    def setweights(self, wih, who):
        self.wih = wih
        self.who = who

    # 神经网络学习训练
    def train(self, inputs_list, targets_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入标签转化成二维矩阵
        targets = np.array(targets_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐层误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新隐层与输出层之间的权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 更新隐层与输出层之间的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
    # 神经网络测试
    def test(self, inputs_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def insertNum():
    retNum = int(numStr.get())
    if(retNum >= 0 and retNum <= 9):
        train_picture(retNum)
        text = tk.StringVar()
        messagebox.showinfo('训练完成','标签为: '+str(retNum)+'\n训练完成')
        numWindow.destroy()
    else:
        messagebox.showerror('输入范围出错','输入不在0到9之间')

def initNum():
    numStr.set('')

def testNum(content):
    return content.isdigit()

def numberInputWindow():
    global numWindow
    numWindow = tk.Toplevel(root)
    #numWindow.geometry('300x300')
    numWindow.title('numberInput')

    text = tk.StringVar()
    text.set('当前用于训练的文件路径是: ' + path + '\n请输入这张图片的数字作为训练标签：')
    textLabel = tk.Label(numWindow, 
                        textvariable = text, 
                        font = ('宋体',14),
                        justify = tk.LEFT,
                        wraplength=450,
                        width = 45)
    textLabel.pack(expand=True,fill="both",padx=5,pady=5)

    global numStr
    numStr = tk.StringVar()

    testCMD = numWindow.register(testNum)
    numEntry = tk.Entry(numWindow,
                        font = ('宋体',16),
                        justify = tk.CENTER,
                        textvariable = numStr,
                        validate="key",
                        validatecommand=(testCMD,'%P'))
    numEntry.pack(expand=True,fill="both",padx=5,pady=5)

    paned4 = tk.PanedWindow(numWindow)
    paned4.pack(expand=True,fill="both")
    buttonInit = tk.Button(paned4,text='清空',command=initNum,font=('宋体',16))
    buttonInit.pack(expand=True,fill="both",padx=5,pady=5,side=tk.LEFT)
    buttonInsert = tk.Button(paned4,text='确定',command=insertNum,font=('宋体',16))
    buttonInsert.pack(expand=True,fill="both",padx=5,pady=5,side=tk.RIGHT)

# 训练单张图片
def train_picture(number):
    
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率
    learning_rate = 0.1
    # 训练次数
    epochs = 5
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取权重存档
    wih = np.loadtxt(open('weights/wih_n.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('weights/who_n.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)

    # 读取待训练的图片
    img = Image.open(path).convert('L')

    inputs = img.resize((28, 28), Image.ADAPTIVE)
    inputs = np.asfarray(inputs) / 255.0 * 0.99 + 0.01
    inputs = inputs.flatten()
    '''
    while(1):
        messagebox.showinfo('提示','在命令行输入该图片的数字作为训练标签\
        \n\n用于训练的文件路径是: '+path)
        print('\n提示:该用于训练的图片路径是: '+path)
        number = int(input('输入这张图片的数字作为训练标签：'))
        if(number >= 0 and number <= 9):
            break
        else:
            messagebox.showerror('输入范围出错','输入不在0到9之间')
            print('输入不在0到9之间\n')
    '''
    targets = np.zeros(output_nodes) + 0.01
    targets[number] = 0.99
    
    for e in range(epochs):
        n.train(inputs, targets)

    np.savetxt('weights/who_n.csv', n.who, delimiter=',')
    np.savetxt('weights/wih_n.csv', n.wih, delimiter=',')
    print('\n用于训练的文件路径是: ' + path)
    print('标签为:' + str(number) + ' 训练完成')
    #messagebox.showinfo('提示','训练完成\n\n用于训练的文件路径是: '+path)


# 手写数字图像识别
def recognition():
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取待预测图片
    img = Image.open(path).convert('L')
    newimg = img.resize((28, 28), Image.ADAPTIVE)
    # newimg.save('picture/2_resize.jpg')
    test_pic = np.array(newimg)

    # 利用神经网络预测
    wih = np.loadtxt(open('weights/wih_n.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('weights/who_n.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    # 正常255表示白色，但mnist数据集255表示黑色，所以现实图片颜色应该翻转一下
    #results = n.test(np.asfarray(255.0 - test_pic.flatten()) / 255.0 * 0.99 + 0.01)
    results = n.test(np.asfarray(test_pic.flatten() / 255.0 * 0.99 + 0.01))
    pre_label = np.argmax(results)
    print('\n识别图片的路径:'+path)
    print('识别结果：', pre_label)
    messagebox.showinfo('识别完成','识别结果: '+str(pre_label)+'\n\n该文件的路径是: '+path)

    return pre_label
    #print(results)

def recognitionCurrent():
    global path
    path = "image.png"
    cv2.imwrite(path , imgArray, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    ret = recognition()

def trainCurrent():
    global path
    path = 'image.png'
    cv2.imwrite(path , imgArray, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    numberInputWindow()

def recognitionByPath():
    global path
    path = filedialog.askopenfilename()
    if len(path) > 0:
        retVal = recognition()

def reset():
    cv2.rectangle(imgArray, (0,0), (500,500), (0,0,0), -1)
    displayImg()

def trainByPath():
    global path
    path = filedialog.askopenfilename()
    if len(path) > 0:
        numberInputWindow()

def displayImg():
    cv2image = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGBA)
    current_image = Image.fromarray(cv2image)
    tkImg = ImageTk.PhotoImage(image = current_image)
    photoLabel.imgtk = tkImg
    photoLabel.config(image=tkImg)

def drawInput(event):
    #print("点击位置:",event.x,event.y)
    #cv2.waitKey(1000)
    cv2.circle(imgArray, (event.x,event.y), 30, (255,255,255), -1)
    displayImg()
    #root.after(0.0001, drawInput)

global imgArray
imgArray = np.zeros((500,500,3),np.uint8)

drawing=False
ix,iy=-1,-1

root = tk.Tk()
root.title('HandwritingRecognition')

photoLabel = tk.Label(root,bg='black')
photoLabel.bind('<Button-1>',drawInput)
photoLabel.bind('<B1-Motion>',drawInput)
photoLabel.pack()
root.config(cursor="arrow")

displayImg()

buttonReset = tk.Button(root,text='重置手写输入窗口',command=reset,font=('宋体',16))
buttonReset.pack(expand=True,fill="both",padx=5,pady=5)

paned1 = tk.PanedWindow(root)
paned1.pack(expand=True,fill="both")

paned2 = tk.PanedWindow(root)
paned2.pack(expand=True,fill="both")

paned3 = tk.PanedWindow(root)
paned3.pack(expand=True,fill="both")

buttonRecognition = tk.Button(paned1,text='识别当前手写数字',command=recognitionCurrent,font=('宋体',16))
paned1.add(buttonRecognition)
buttonRecognition.pack(side=tk.LEFT,expand=True,fill="both",padx=5,pady=5)

buttonTrainCurrent = tk.Button(paned1,text='训练当前手写数字',command=trainCurrent,font=('宋体',16))
paned1.add(buttonTrainCurrent)
buttonTrainCurrent.pack(side=tk.RIGHT,expand=True,fill="both",padx=5,pady=5)

buttonRecognitionByPath = tk.Button(paned2,text='选择路径识别图片',command=recognitionByPath,font=('宋体',16))
paned2.add(buttonRecognitionByPath)
buttonRecognitionByPath.pack(side=tk.LEFT,expand=True,fill="both",padx=5,pady=5)

buttonTrainByPath = tk.Button(paned2,text='选择路径训练图片',command=trainByPath,font=('宋体',16))
paned2.add(buttonTrainByPath)
buttonTrainByPath.pack(side=tk.RIGHT,expand=True,fill="both",padx=5,pady=5)

root.mainloop()
cv2.destroyAllWindows()