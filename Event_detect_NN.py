# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:57:11 2019

 --- 10.10以Kears处理路透社新闻数据为模板，修改程序实现了NN方式对事件类型进行分类

@author: ThinkPad
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from prettytable import PrettyTable

#path = 'C:\\Users\\ThinkPad\\Desktop\\iris.data'  # 数据文件路径
#path = 'C:\\Users\\ThinkPad\\Desktop\\工作记录\\20190923\\csv2xlsx\\事件状态统计\\1.txt'
path = 'C:\\Users\\ThinkPad\\Desktop\\工作记录\\1EE车队样本数据\\sample11.15\\EE_加入raw_y_SVM_NN_11.15.txt'

def event_type(s):
    it = {b'1': 0, b'2': 1, b'3': 2, b'4': 3}
    return it[s]
#def event_type(s):
#    it = {b'0': 0, b'1': 1, b'2': 2}
#    return it[s]
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
def div(x,y):
    try:
        return round(x/y,3)
    except ZeroDivisionError:
        return 0
def draw(x_test, y_test):
    out = model.predict_classes(x_test) 
    r1_1=r1_2=r1_3=r1_4=r2_1=r2_2=r2_3=r2_4=r3_1=r3_2=r3_3=r3_4=r4_1=r4_2=r4_3=r4_4 = 0
    for i in range(len(x_test)):
        label = str(int(y_test[i][0]))
        out_label = str(int(out[i]))
        if label == '0':
            if  out_label == '0':
                r1_1 += 1
            elif out_label == '1':
                r1_2 += 1
            elif out_label == '2':
                r1_3 += 1
            elif out_label == '3':
                r1_4 += 1 
        if label == '1':
            if  out_label == '0':
                r2_1 += 1
            elif out_label == '1':
                r2_2 += 1
            elif out_label == '2':
                r2_3 += 1
            elif out_label == '3':
                r2_4 += 1 
        if label == '2':
            if  out_label == '0':
                r3_1 += 1
            elif out_label == '1':
                r3_2 += 1
            elif out_label == '2':
                r3_3 += 1
            elif out_label == '3':
                r3_4 += 1 
        if label == '3':
            if  out_label == '0':
                r4_1 += 1
            elif out_label == '1':
                r4_2 += 1
            elif out_label == '2':
                r4_3 += 1
            elif out_label == '3':
                r4_4 += 1 
    r1 = r1_1 + r1_2 + r1_3 + r1_4
    r2 = r2_1 + r2_2 + r2_3 + r2_4
    r3 = r3_1 + r3_2 + r3_3 + r3_4
    r4 = r4_1 + r4_2 + r4_3 + r4_4 
    print("总的成功率： ", round( (r1_1+r2_2+r3_3+r4_4)/(r1+r2+r3+r4) ,3))
    x= PrettyTable(["真实\预测", "急转", "急刹", "急加", "正常", "累计", "检测成功率"])
    x.add_row(["急转",r1_1,r1_2,r1_3,r1_4,r1, div(r1_1,r1)])
    x.add_row(["急刹",r2_1,r2_2,r2_3,r2_4,r2, div(r2_2,r2)])
    x.add_row(["急加",r3_1,r3_2,r3_3,r3_4,r3, div(r3_3,r3)])
    x.add_row(["正常",r4_1,r4_2,r4_3,r4_4,r4, div(r4_4,r4)])
    print(x)

if __name__ == '__main__':    
    data = np.loadtxt(path, dtype=float, delimiter=' ', converters={15: event_type})
    x, y = np.split(data, (15,), axis=1)
    
    # --- 拓展样本，在原来15维基础上，增加了平方项15维，交叉乘积项105维，共135
    # 拓展平方项
    x_extend_square = copy.deepcopy(x)
    for i in range(len(x_extend_square)):
        for j in range(len(x_extend_square[0])):
            x_extend_square[i][j] = (x_extend_square[i][j])**2
    # 拓展交叉项        
    x_extend_cross = []
    for i in range(len(x)):
        x_extend_cross.append([])
        for j in range(len(x[0])-1):
            for k in range(j+1):
                x_extend_cross[i].append(x[i][j] * x[i][k])
    x_extend_cross = np.array(x_extend_cross)    
    x = np.hstack((x,x_extend_square))
    x = np.hstack((x,x_extend_cross))
    
    # --- 分离训练样本和测试样本
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=None,train_size=0.5)
    
    # --- 对label进行独热编码
    one_hot_train_labels = to_categorical(y_train)
    one_hot_test_labels = to_categorical(y_test)
    
    # --- 模型搭建
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(135,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    #model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # 分离验证集
    x_val = x_train[:20]
    partial_x_train = x_train[20:]
    y_val = one_hot_train_labels[:20]
    partial_y_train = one_hot_train_labels[20:]
    #x_val = x_train[:30]
    #partial_x_train = x_train[30:]
    #y_val = one_hot_train_labels[:30]
    #partial_y_train = one_hot_train_labels[30:]
    # 训练模型
    start = time.process_time()
    # 设置学习率,每100个epoch学习率下降为原来的10%
    reduce_lr = LearningRateScheduler(scheduler)
    history = model.fit(partial_x_train,partial_y_train,epochs=30, batch_size=40, callbacks=[reduce_lr], validation_data=(x_val,y_val))
    print("总时间：", time.process_time() - start)
    
    # ploting the training and validation loss
    loss = history.history['loss']
    val_loss  = history.history['val_loss']
    epochs = range(1,len(loss)+1)
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validating loss')
    plt.title('Training and Validating loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
     
    #ploting the training and validation accuracy
    #plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1,2,2)
    plt.plot(epochs,acc,'ro',label='Training acc')
    plt.plot(epochs,val_acc,'r',label='Validating acc')
    plt.title('Training and Validating accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    #evaluate
    final_result = model.evaluate(x_test,one_hot_test_labels)
    print(final_result)
    print("训练样本比例：",round(len(y_train)/len(y),3))
    draw(x_test, y_test)


















#二分类问题
#from sklearn.datasets import load_breast_cancer
#from sklearn.model_selection import train_test_split
#from tensorflow import keras
#import tensorflow as tf
#from tensorflow.keras import layers
#whole_data = load_breast_cancer()
#x_data = whole_data.data             #(569, 30)
#y_data = whole_data.target           #(569,)二分类
#
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=7)
#
##print(x_train.shape, ' ', y_train.shape)
##print(x_test.shape, ' ', y_test.shape)
#
## 构建模型
#model = keras.Sequential([
#    layers.Dense(32, activation='relu', input_shape=(30,)),
#    layers.Dense(32, activation='relu'),
#    layers.Dense(1, activation='sigmoid')
#])
#
#model.compile(optimizer=keras.optimizers.Adam(),
#             loss=keras.losses.binary_crossentropy,
#             metrics=['accuracy'])
#model.summary()
#
#model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
#
#model.evaluate(x_test, y_test)
#
##print(model.metrics_names)



# --- 动态画图
#import matplotlib.pyplot as plt
#fig,ax=plt.subplots()
#y1=[]
#for i in range(50):
#    y1.append(i)
#    ax.cla()
#    ax.bar(y1,label='test',height=y1,width=0.3)
#    ax.legend()
#    plt.pause(0.3)
#其中y1是数据的Y值，只要不停地更y1的数组内容，就可以0.3S刷新一次

##数据类型改变
#x=[1,2,-2,2,2,3,5,-4,4,]
#y = sorted(x, key=lambda x:x**2)
#z=list(map(str,x))
#a=[1,2,3]
#b=[4,5,6]
#
############################################3
##draw 2D
#import numpy as np
#import pylab as pl
#times=np.arange(0,1,0.1)  #times为x的值，0为起点，5为终点，0,01为步长
#fun=lambda x: 1/x**1.25 - 1/x + 1#fun为关于x的函数，也就是对应于x的y的值
#
#pl.plot(times,fun(times))  #画图
#pl.xlabel(u"pct")  #x轴的标记
#pl.ylabel("threshold ratio")  #y轴的标记
#pl.title("ratio")  #图的标题
#pl.show()
#
#x = range(10)
#y =[i*i for i in x]
#pl.plot(x,y,'ob-')
#pl.show