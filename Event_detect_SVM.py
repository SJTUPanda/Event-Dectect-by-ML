# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:25:20 2019

@author: ThinkPad
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

def event_type(s):
    it = {b'1': 1, b'2': 2, b'3': 3, b'4': 4}
    return it[s]
def extend(a, b, r):
    x = b - a
    m = (a + b) / 2
    return m-r*x/2, m+r*x/2

#path = 'C:\\Users\\ThinkPad\\Desktop\\iris.data'  # 数据文件路径
path = 'C:\\Users\\ThinkPad\\Desktop\\工作记录\\1EE车队样本数据\\sample11.15\\EE_加入raw_y_SVM_NN_11.15.txt'
#path = 'C:\\Users\\ThinkPad\\Desktop\\工作记录\\20190923\\csv2xlsx\\事件状态统计\\1.txt'
data = np.loadtxt(path, dtype=float, delimiter=' ', converters={15: event_type})


x, y = np.split(data, (15,), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=None,train_size = 0.5)

clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
#clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
#kernel=’linear’时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。 
#kernel=’rbf’时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合
clf.fit(x_train, y_train.ravel())

print('数据总数： ',len(y))
print(' --- 急转弯样本数: ',list(y).count([1.]))
print(' --- 急刹车样本数: ',list(y).count([2.]))
print(' --- 急加速样本数: ',list(y).count([3.]))
print(' --- 正常样本数: ',list(y).count([4.]))
#print('训练样本比例: ',train_size*100,'%')
print('训练样本比例：',len(x_train)/len(y))
print('训练数据的判断精度： ',round(clf.score(x_train, y_train),3))  # 精度
y_hat = clf.predict(x_train)

print('测试数据的判断精度： ',round(clf.score(x_test, y_test),3))


y_hat2 = clf.predict(x_test)
r1_1=r1_2=r1_3=r1_4=r2_1=r2_2=r2_3=r2_4=r3_1=r3_2=r3_3=r3_4=r4_1=r4_2=r4_3=r4_4 = 0

def div(x,y):
    try:
        return round(x/y,3)
    except ZeroDivisionError:
        return 0

for i in range(len(x_test)):
    label = str(int(y_test[i][0]))
    out_label = str(int(y_hat2[i]))
    if label == '1':
        if  out_label == '1':
            r1_1 += 1
        elif out_label == '2':
            r1_2 += 1
        elif out_label == '3':
            r1_3 += 1
        elif out_label == '4':
            r1_4 += 1 
    if label == '2':
        if  out_label == '1':
            r2_1 += 1
        elif out_label == '2':
            r2_2 += 1
        elif out_label == '3':
            r2_3 += 1
        elif out_label == '4':
            r2_4 += 1 
    if label == '3':
        if  out_label == '1':
            r3_1 += 1
        elif out_label == '2':
            r3_2 += 1
        elif out_label == '3':
            r3_3 += 1
        elif out_label == '4':
            r3_4 += 1 
    if label == '4':
        if  out_label == '1':
            r4_1 += 1
        elif out_label == '2':
            r4_2 += 1
        elif out_label == '3':
            r4_3 += 1
        elif out_label == '4':
            r4_4 += 1 
r1 = r1_1 + r1_2 + r1_3 + r1_4
r2 = r2_1 + r2_2 + r2_3 + r2_4
r3 = r3_1 + r3_2 + r3_3 + r3_4
r4 = r4_1 + r4_2 + r4_3 + r4_4

print()
print("总的成功率： ", round( (r1_1+r2_2+r3_3+r4_4)/(r1+r2+r3+r4) ,3))
x= PrettyTable(["真实\预测", "急转", "急刹", "急加", "正常", "累计", "检测成功率"])
x.add_row(["急转",r1_1,r1_2,r1_3,r1_4,r1, div(r1_1,r1)])
x.add_row(["急刹",r2_1,r2_2,r2_3,r2_4,r2, div(r2_2,r2)])
x.add_row(["急加",r3_1,r3_2,r3_3,r3_4,r3, div(r3_3,r3)])
x.add_row(["正常",r4_1,r4_2,r4_3,r4_4,r4, div(r4_4,r4)])
print(x)

#
#x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
#x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
#x1_min, x1_max = extend(x1_min, x1_max, 1.05)#以1.05的系数减小最小值，增大最大值
#x2_min, x2_max = extend(x2_min, x2_max, 1.05)
#    
#x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
#grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
#
#new_array = np.zeros((40000,13))
#for i in range(40000):
#    new_array[i] = x_train[0][2:]
#
#
#grid_test = np.hstack((grid_test,new_array))
##for i in range(len(grid_test)):
##    grid_test[i] = np.array(list(grid_test[i]) + list(x_train[0][2:]))
#  
#    
##mpl.rcParams['font.sans-serif'] = [u'SimHei']
##mpl.rcParams['axes.unicode_minus'] = False
##
##cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
##cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
#
#grid_hat = clf.predict(grid_test)  # 预测分类值
#grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
##alpha = 0.5
##plt.figure(facecolor='w')
##plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
##plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, markeredgecolor='k')
#
#cm_light = mpl.colors.ListedColormap(['#FF8080', '#A0FFA0', '#6060FF'])
#cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b'])
#mpl.rcParams['font.sans-serif'] = [u'SimHei']
#mpl.rcParams['axes.unicode_minus'] = False
#plt.figure(facecolor='w')
#plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  
#y = y.flatten() 
#plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, alpha=0.7)
#plt.xlim((x1_min, x1_max))
#plt.ylim((x2_min, x2_max))
#plt.grid(b=True)
#plt.tight_layout(pad=2.5)
#plt.xlabel(u'X幅度', fontsize=13)
#plt.ylabel(u'Z幅度', fontsize=13)
#plt.title(u'red:turn   green:brake   blue:accel', fontsize=18)
#plt.show()
#    
#    
##plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, zorder=10)  # 圈中测试集样本
##plt.xlabel(u'花瓣长度', fontsize=13)
##plt.ylabel(u'花瓣宽度', fontsize=13)
##plt.xlim(x1_min, x1_max)
##plt.ylim(x2_min, x2_max)
##plt.title(u'SVM分类', fontsize=15)
##plt.show()
