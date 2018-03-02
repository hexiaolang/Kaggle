#-*- coding:utf-8 -*-

from __future__ import division
import numpy
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC

COMPONENT_NUM = 35 #设置pca降维的维度值

print ('\n---- Reading the trainning dataset ----')
with open('train.csv', 'r') as reader:
    reader.readline()#去掉表头
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
    print ('the size of training dataset is ' + str(len(train_label)))

    print ('\n---- Data Reduction ----')
    train_label = numpy.array(train_label)#将list转换成numpy数组
    train_data = numpy.array(train_data)
    print ("the initial dimension of the training dataset is "+ str(train_data.shape))#原始数据集的维度
    pca = PCA(n_components=COMPONENT_NUM, whiten=True)
    pca.fit(train_data)#Fit the model with X
    train_data = pca.transform(train_data)#Fit the model with X and 在X上完成降维
    print ("after data reduction, the dimension of the training dataset is "+ str(train_data.shape))#降维后数据集的维度

    print ('\n---- Training SVM ----')
    start = time.time()
    svc = SVC()
    svc.fit(train_data, train_label)#训练SVM
    end = time.time()
    print ("the time of training SVM is "+ str(round(end - start, 6))+ "s")

    print ('\n---- Reading the testing data ----')
    with open('test.csv', 'r') as reader:#加载测试集
        reader.readline()
        test_label = []
        test_data = []
        for line in reader.readlines():
            data = list(map(int, line.rstrip().split(',')))
            test_label.append(data[0])
            test_data.append(data[1:])
    print ('the size of testing dataset is ' + str(len(test_data)))

    print ('\n---- Predicting the label of testing data ----')
    test_data = numpy.array(test_data)
    test_data = pca.transform(test_data)
    start = time.time()
    predict = svc.predict(test_data)
    end = time.time()
    print ("the time of predicting testing data is "+ str(round(end - start, 6))+ "s")

    print ('\n---- Saving the predicting results ----')#保存预测结果
    with open('predict.csv', 'w') as writer:
        writer.write('"ImageId","Label"\n')
        count = 0
        for p in predict:
            count +=1
            writer.write(str(count)+',"'+str(p)+'"\n')

    print ('\n---- Evaluating the predicting results ----')#对比预测结果与实际结果
    count = 0
    total = len(test_label)
    for i in range(total):
        if(test_label[i] == predict[i]):
            count +=1
    print ('the number of instances that are correctly classified is ' + str(count))
    print ('the total number of testing dataset is ' + str(total))
    print ('accuracy = ' + str(round(count/total, 6)))

