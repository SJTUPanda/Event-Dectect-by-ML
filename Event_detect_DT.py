# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:15:14 2019

@author: ThinkPad
"""

import pandas as pd
from sklearn import tree
from pandas.core.frame import DataFrame
import pydotplus
from IPython.display import display, Image
from prettytable import PrettyTable
import random 

path = 'C:\\Users\\ThinkPad\\Desktop\\工作记录\\1EE车队样本数据\\sample11.15\\EE_加入raw_y_贝叶斯11.15.txt'
dataset = pd.read_csv(path, dtype=str, delimiter=' ',header=None)

feature_columns = ['y幅度', 'X幅度', 'Z幅度', 'X标准差', 'Z标准差', 'X均值', 'Z均值', 'Z前半均值', 'Z后半均值', 'X最小值', '偏转标准差', '偏转幅度', '前半速度', '后半部速度', '持续时间']
label_column = ['type']

# 修改feature和label的列索引
features = dataset.loc[:,list(dataset.columns)[:15]]
label = dataset.loc[:,list(dataset.columns)[15]]
for i in range(len(feature_columns)):
    features.rename(columns={i:feature_columns[i]}, inplace=True)
label.rename(columns={0:'type'}, inplace=True)


# 测试集正确率统计
def div(x,y):
    try:
        return round(x/y,3)
    except ZeroDivisionError:
        return 0

       


#初始化一个决策树分类器
features = pd.get_dummies(features) # one-hot编码
train_size = 0.5

train_slice = random.sample(list(features.index), int(len(features) * train_size))  #从list中随机获取5个元素，作为一个片断返回  
test_slice = [item for item in list(features.index) if item not in train_slice]

x_train_nd = features.ix[train_slice,]
y_train_nd = label.ix[train_slice,]
x_test_nd = features.ix[test_slice,]
y_test_nd = label.ix[test_slice,]


clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1)
clf = clf.fit(x_train_nd.values, y_train_nd.values)

# 画决策树
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=features.columns,
                                class_names = ['1', '2', '3', '4'],
                                filled = True,
                                rounded =True
                               )
graph = pydotplus.graph_from_dot_data(dot_data.replace('helvetica', '"Microsoft YaHei"'))
display(Image(graph.create_png()))

# 测试集结果
print(clf.score(x_test_nd.values, y_test_nd.values))
x_test = list(x_test_nd.values)
y_test = list(y_test_nd.values)
y_hat2 = list(clf.predict(x_test_nd.values))

r1_1=r1_2=r1_3=r1_4=r2_1=r2_2=r2_3=r2_4=r3_1=r3_2=r3_3=r3_4=r4_1=r4_2=r4_3=r4_4 = 0    
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

















