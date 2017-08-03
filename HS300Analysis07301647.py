# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:13:46 2017

@author: Geng Chen
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

xl = pd.ExcelFile('DataAlpha07302311.xlsx')
df = pd.read_excel(xl, sheet_name = 'Sheet1')#读取数据
pd.set_option('display.unicode.east_asian_width',True)#To align header properly


#t1 = [20090105,20111230,20121231]
t1 = [20100104,20121231,20131231]
t2 = [20110104,20131231,20141231]
t3 = [20120104,20141231,20151231]
t  = [t1,t2,t3]



#helper method
alphas_list = ['','x1(6)',	'x2(12)',	'x3(23)',	'x4(53)',	'x5(54)',	'x6(101)','x7(9)',	'x8(21)',	'x9(46)',
                'x10(49)','x11(28)','x12(102)','x13(6)','x14(9)']

alphas_index = [5] #在这里输入单独Alphas值
alpha_b = alphas_list[1] #在这里输入连续Alphas的起始值
alpha_e = alphas_list[2] #在这里输入连续Alphas的


alphas = df.loc[:,alpha_b:alpha_e]#Feature Matrix
    

labels = df.loc[:,'Label']#截取胜负值的dataframe--->lable matrix

a = pd.Index(df['TDATE']).get_loc(20121231) + 2 #需要设置 
b = pd.Index(df['TDATE']).get_loc(20151231) + 1 #需要设置
df2 = df.loc[a:b]
df2 = df2.assign(预测 = np.nan)

 
for p in t:
    b = pd.Index(df['TDATE']).get_loc(p[0])
    e = pd.Index(df['TDATE']).get_loc(p[1])
    h = pd.Index(df['TDATE']).get_loc(p[2]) #回测周期的最后一天的index
    training_set = alphas.loc[b:e].values
    targets_set = labels.loc[b+1:e+1].values
    clf1 = svm.SVC(gamma=1/65, C=30.)
##    clf2 = tree.DecisionTreeRegressor()
#    clf3 = tree.DecisionTreeClassifier()
#    clf4 = KNeighborsClassifier(n_neighbors = 5)   
##    clf5 = linear_model.Ridge (alpha = .05)
##    clf6 = linear_model.LinearRegression()
##    clf7 = linear_model.Lasso(alpha = 0.1)
##    clf8 = linear_model.BayesianRidge()
#    clf9 = LogisticRegression(C=100, penalty='l2', tol=2)
#    clf10 = SGDClassifier(loss="hinge", penalty="l1")  
#    clf11 = RandomForestClassifier(n_estimators=10,max_depth=None, min_samples_split=2, random_state=0,bootstrap = True)
#    clf12 = AdaBoostClassifier(n_estimators=1000)
#    clf13 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
#    eclf = VotingClassifier(estimators=[('1', clf1), ('3', clf3), ('4', clf4),('9',clf9),('10',clf10),('11',clf11),('12',clf12),('13',clf13)], voting='hard')
    
    clf1.fit(training_set, targets_set)
#    clf10.fit(training_set, targets_set)
#    clf3.fit(training_set, targets_set)
#    clf4.fit(training_set, targets_set)
#    clf9.fit(training_set, targets_set)
#    clf10.fit(training_set, targets_set)
#    clf11.fit(training_set, targets_set)
#    clf12.fit(training_set, targets_set)
#    clf13.fit(training_set, targets_set)
#    eclf.fit(training_set, targets_set)
    

    predicted = pd.DataFrame(clf1.predict(alphas.loc[e+1:h].values),
                         index = df.index[e+2:h+2],columns=['预测'])#index列表的最后一个值是取不到的，所以h+2而不是h+1
    df2.update(predicted)


df2 = df2.assign(预测正确 = np.nan,累计净值 = np.nan, 回撤率 = np.nan, 单日盈亏 = np.nan,大盘累计净值=np.nan)
capital = 1
capital2 = 1
for i in df2.index:
    capital2 *= 1+df2.loc[i,'涨跌%']#跟踪大盘
    df2.set_value(i,'大盘累计净值',capital2)#跟踪大盘
    if df2.loc[i,'Label'] == df2.loc[i,'预测']:
        df2.set_value(i, '预测正确', 1)
    else:
        df2.set_value(i, '预测正确', 0)
    
    if df2.loc[i,'预测正确'] == 1:
        c = capital*abs(df2.loc[i,'涨跌%'])#累计净值变化
        df2.set_value(i, '单日盈亏', c) 
        capital += c
        df2.set_value(i, '累计净值', capital)
    else:
        c = capital*abs(df2.loc[i,'涨跌%'])#累计净值变化
        df2.set_value(i, '单日盈亏', -c) #其实做空是可以加杠杆的，这里假设没有杠杆
        capital -= c
        df2.set_value(i, '累计净值', capital)


p = df2[df2.单日盈亏 > 0].单日盈亏.sum() #总盈
l = df2[df2.单日盈亏 < 0].单日盈亏.sum() #总亏
p_l_ratio = -p/l #盈亏比
annualized_return = (df2.tail(1).累计净值.item()-1)/len(df2)*365
prediction_accuracy = df2.预测正确.sum()/len(df2)#胜率
#计算最大回撤率
for i in df2.index:
    c = df2.loc[i,'累计净值']
    df2.set_value(i, '回撤率', (c-df2.loc[i:,'累计净值'].min())/c)
max_drawdown = df2.loc[:,'回撤率'].max()#最大回撤率
ACCNAV = df2.tail(1).累计净值.item()#期末累计净值
sr = ((1+df2.单日盈亏.mean())**242-1-0.035)/(df2.单日盈亏.std()*242**0.5)

result_index = ['最大回撤率: ', '年化收益率：', 
                '盈亏比: ', '胜率: ', '累计净值: ','夏普比率']
result_data = [max_drawdown, annualized_return, p_l_ratio,
               prediction_accuracy, ACCNAV, sr]
result_table = pd.Series(result_data, result_index)
print()
print('     Analysis Result')
print(result_table)

plt.rcParams['font.sans-serif']=['SimHei']
#plt.figure(figsize=(16,8))
plt.figure(figsize=(20,10 )) 
y1 = df2.累计净值.values
y2 = df2.大盘累计净值.values
x = df2.index.values
plt.plot(x,y1,label =  "SVM策略")
plt.plot(x,y2, label =  "沪深300")
plt.xlabel('时间',fontsize=20)
plt.ylabel('累计净值',fontsize=20) 
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1.,prop={'size': 20})
#plt.savefig("RF Chart.jpg") 
plt.show()



#writer = pd.ExcelWriter('prediction result.xlsx', engine='xlsxwriter')
#df2.to_excel(writer, sheet_name='Sheet1')
#writer.save()