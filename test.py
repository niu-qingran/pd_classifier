# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:44:05 2019

@author: win10
"""

import csv
import os
from math import sqrt
import numpy as np
import scipy.signal as signal
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pylab as pl
import pywt
#,'ZCN[1]','ZCN[2]'
column_name1=['range[0]','max[1]','mean[1]','mean_abs[1]','RMS[1]','STD[1]','skew[1]',
              'ZCN[1]','CoV[1]']
column_name2=['range[1]','max[2]','mean[2]','mean_abs[2]','RMS[2]','STD[2]','skew[2]',
              'ZCN[2]','CoV[2]']


def load_data(path):
    os.chdir(path)
    #files=os.listdir(path)
    data=[]
    for file in glob.glob("*.csv"): #遍历文件夹
        meta_data=pd.read_csv(file,skiprows=range(0, 2))
        data.append(meta_data) #每个文件的文本存到list中
    #data=[create_data(file) for file in glob.glob("*.csv")]
    return data

def band_filter(data):
        b,a=signal.butter(8,[0.005,0.6],'bandpass')
        buffer_data=signal.filtfilt(b,a,data)
        return buffer_data
    
def wavelet_dec(data):
        db4=pywt.Wavelet('db4')
        A5,D5,D4,D3,D2,D1= pywt.wavedec(data,db4,mode='symmetric',level=5)
        D5=np.zeros(D5.shape[0])
        D4=np.zeros(D4.shape[0])
        D3=np.zeros(D3.shape[0])
        D2=np.zeros(D2.shape[0])
        D1=np.zeros(D1.shape[0])
        data_rec=pywt.waverec([A5,D5,D4,D3,D2,D1],db4)
        return data_rec
    
def data_pre(data):
    b=[]
    meta=['Acce_0','Gyro_0','Acce_1','Gyro_1','Acce_2','Gyro_2']
    for i in data:
        a=pd.DataFrame(columns=meta)
        for j in meta:   
            a[j]=wavelet_dec(i[j])
        b.append(a)
    return b

def merge_data(i,item1,item2,item3):
    merge=i[item1]**2+ i[item2]**2+ i[item3]**2
    return np.sqrt(merge)

def data_merged(raw_data):
    data=[]   
    #for i in data_pre(raw_data):
    for i in raw_data:
        a=pd.DataFrame(columns=['Acce_0','Gyro_0','Acce_1','Gyro_1','Acce_2','Gyro_2'])
        Acce_0=merge_data(i,meta[0],meta[1],meta[2])
        Gyro_0=merge_data(i,meta[3],meta[4],meta[5])
        Acce_1=merge_data(i,meta[6],meta[7],meta[8])
        Gyro_1=merge_data(i,meta[9],meta[10],meta[11])
        Acce_2=merge_data(i,meta[12],meta[13],meta[14])
        Gyro_2=merge_data(i,meta[15],meta[16],meta[17])
        a['Acce_0']=Acce_0
        a['Acce_1']=Acce_1
        a['Acce_2']=Acce_2
        a['Gyro_0']=Gyro_0
        a['Gyro_1']=Gyro_1
        a['Gyro_2']=Gyro_2
        data.append(a)
    #data=data_pre(data)
    return data

def ext_fea(item):
    return [np.ptp(item),np.max(item),np.mean(item),np.mean(abs(item)),
            mse(item),np.std(item),item.skew(),calZeroCrossingNum(item),
            np.std(item)/np.mean(item)]

def segment(data,column):
    data=data[column]-data[column][0:100].mean()
    data_mid=signal.medfilt(abs(data),251)
    start=[]
    end=[]
    low=1
    high=0
    for i in range(len(data_mid)):
        if low==1:
            if data_mid[i]>0.6:
                start.append(i)
                low=0
                high=1
                continue
        if high==1:
            if data_mid[i]<0.4:
                end.append(i)
                high=0
                low=1
                continue

    drop_i=[]
    for i in range(len(start)):
        #print(i)
        if end[i]-start[i]<500:
            drop_i.append(i)
    
    for i in drop_i:
        start[i]=0
        end[i]=0
    #start.remove(0)
    #end.drop(0)
    for i in range(len(drop_i)):
        start.remove(0)
        end.remove(0)
    start=[i-120 for i in start]
    end=[i+100 for i in end]
    return start,end
#
#    feature=pd.DataFrame(columns=[column_name1])
#    for i in range(len(start)):
#        feature.loc[i]=(ext_fea(data[start[i]:end[i]]))
#    m.plot()
#    for i in range(len(start)):        
#        data[start[i]:end[i]].plot()              
##    return [feature['max[1]'].mean(),feature['mean[1]'].mean(),
##            feature['Varience[1]'].mean(),feature['STD[1]'].mean()]
#    return [feature[col].mean() for col in feature.columns]

def seg_fea(start,end,data,fea):
    feature=pd.DataFrame(columns=column_name1)
    data=data[fea]-data[fea][0:100].mean()
    for i in range(len(start)):
        feature.loc[i]=(ext_fea(data[start[i]:end[i]]))
    return [feature[col].mean() for col in feature.columns]

#def ext_fea(item,feature):
#    return [np.max(item[feature]),np.mean(item[feature]),mse(item[feature]),
#            np.std(item[feature])**2,np.std(item[feature])]
#    
def make_dataset(data):
    feature1=pd.DataFrame(columns=column_name1)
    feature2=pd.DataFrame(columns=column_name2)   
    for i,item in enumerate(data):
        start,end=segment(item,'Acce_1')
        feature1.loc[i]=seg_fea(start,end,item,'Acce_1')   
        #print(i)
        feature2.loc[i]=seg_fea(start,end,item,'Gyro_1')
    feature=pd.concat([feature1,feature2],axis=1)
    return feature

def mse(data):
    SUM=0
    for i in data:
        SUM=SUM+i*i
    return sqrt(SUM/len(data))


def neg_label(num):
    a=np.ones(num)
    for i in range(num):
        a[i]=-1
    return a

def pos_label(num):
    a=np.ones(num)
    for i in range(num):
        a[i]=1
    return a
  
def data_made(path):
    data_meta=load_data(path)
    data_filt=data_merged(data_meta)
    data=data_pre(data_filt)
    return data_meta,data_filt,data,make_dataset(data)

def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0   

def calZeroCrossingNum(data) :
    data=data.reset_index(drop=True)
    SUM = 0
    for i in range(len(data)-1) :
        SUM = SUM + np.abs(sgn(data[i+1]) - sgn(data[i]))
    return SUM


if __name__=="__main__":
    #print(os.getcwd()) # 打印当前工作目录    
    #导入病人数据
    meta=['Acce_x[0]','Acce_y[0]','Acce_z[0]','Gyro_x[0]','Gyro_y[0]','Gyro_z[0]',
          'Acce_x[1]','Acce_y[1]','Acce_z[1]','Gyro_x[1]','Gyro_y[1]','Gyro_z[1]',
          'Acce_x[2]','Acce_y[2]','Acce_z[2]','Gyro_x[2]','Gyro_y[2]','Gyro_z[2]']

    path1=r'C:\Users\win10\Desktop\JOB\PD2019.1.8__8'
    PDdata_meta,PDdata_filt,PDdata,PDfeature=data_made(path1)
    #导入健康人数据v  
    path2=r'C:\Users\win10\Desktop\JOB\HP2019.1.9__10'
    HPdata_meta,HPdata_filt,HPdata,HPfeature=data_made(path2)
    #查看原始数据
    path3=r'C:\Users\win10\Desktop\JOB\PD2019.1.15__10'
    TEdata_meta,TEdata_filt,TEdata,TEfeature=data_made(path3)
    
    
    feature=pd.concat([PDfeature,HPfeature,TEfeature],axis=0,ignore_index=True)
    a=neg_label(len(PDdata))
    b=pos_label(len(HPdata))
    c=neg_label(len(TEdata))
    labels=np.concatenate((a,b,c))
    feature['label']=labels
    #feature=feature.drop(index=20)
#    from sklearn.utils import shuffle
#    feature=shuffle(feature)
#    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature.iloc[:,0:-1])
    
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test =train_test_split(scaled,
                            feature.iloc[:,-1],test_size=0.4, random_state=0)
#    
#    X_train=feature.iloc[:,0:-1]
#    y_train=labels
#    #y_train=feature.iloc[:,-1]
#       
#    X_test=TEfeature
#    y_test=neg_label(len(TEdata))
#    
#   
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
    """=============第一个模型 线性SVM=================""" 
    from sklearn.svm import SVC
    
    #clf = SVC(kernel='linear',C=0.1)
    #clf=SVC(kernel='poly',degree=3,gamma=0.5,coef0=0)
    clf=SVC(C=1, gamma=0.05)
    clf.fit(X_train,y_train)
    pred_y = clf.predict(X_test)
    print(classification_report(y_test,pred_y))
    confusion_matrix=confusion_matrix(y_test,pred_y)
    
       
    from sklearn.model_selection import GridSearchCV
    
    grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1,0.05, 0.005]}, cv=4)
    grid.fit(X_train, y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    
    
    """=============第二个模型 adaboost================="""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=20, learning_rate=0.5)
    bdt.fit(X_train,y_train)
    pred_y = bdt.predict(X_test)
    print(classification_report(y_test,pred_y))
    
    """=============第三个模型 逻辑回归================="""
    from sklearn.linear_model.logistic import LogisticRegression
    classifier=LogisticRegression()
    classifier.fit(X_train,y_train)
    pred_y = classifier.predict(X_test)
    print(classification_report(y_test,pred_y))
    #confusion_matrix=confusion_matrix(y_test,pred_y)
    
    """==============第四个模型  ELM============================"""
#    from hpelm import ELM
#    elm=ELM(10,18)
#    elm.add_neurons(50,'sigm')
#    elm.train(np.mat(X_train), np.mat(y_train), "LOO")
#    y_pred=elm.predict(X_test)
    
    """=============第五个模型 LDA================="""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    
    
    