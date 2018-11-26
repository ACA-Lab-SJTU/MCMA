
# coding: utf-8

# In[1]:

import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import os

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

from keras.models import Sequential,model_from_json
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD,RMSprop,Adamax,Adam,Adagrad
from keras.losses import mean_absolute_percentage_error
from keras.utils import np_utils,generic_utils
from keras import metrics
import error
import math
import multiprocessing as mp
import json
import sys


# In[2]:

#Parametres
app_name='bessel_Jnu'
numA=1
iteration=1
epochA=20
epochC=15
batch_size=128
eb=[]
net_A=[]
net_C=[]
lenN=0
method=''

#error_type="absolute_error"
error_type="rmse"
#error_type="relative_error"
def error_compare(error_type):
    if (error_type=="absolute_error"):
        print(1)
        return error.absolute_error
    if (error_type=="rmse"):
        print(2)
        return error.rmse
    if (error_type=="relative_error"):
        print(3)
        return error.relative_error
error_measure=error_compare(error_type)

def get_net():
    global net_A
    global net_C
    if (app_name=='fft'):
        net_A=[1,2,2,2]
        net_C=[1,2,numA+1]
    if (app_name=='bessel_Jnu'):
        net_A=[2,4,4,1]
        net_C=[2,4,numA+1]
    if (app_name=='blackscholes'):
        net_A=[6,8,1]
        net_C=[6,8,numA+1]
    if (app_name=='jmeint'):
        net_A=[18,32,16,2]
        net_C=[18,16,numA+1]
    if (app_name=='jpeg'):
        net_A=[64,16,64]
        net_C=[64,18,numA+1]
    if (app_name=='inversek2j'):
        net_A=[2,8,2]
        net_C=[2,8,numA+1]
    if (app_name=='sobel'):
        net_A=[9,8,1]
        net_C=[9,8,numA+1]
    if (app_name=='kmeans'):
        net_A=[6,8,4,1]
        net_C=[6,8,4,numA+1]


# In[3]:

#Input Output processing
def format_data(data):
    try:
        if (len(data.shape)==1):
            return data.reshape((data.shape[0],1))
        else:
            return data
    except:
        print("Error! data is not a numpy object,format_data failed!")
        exit(0)

def load_data(app_name):
    x_train=np.loadtxt('../data/'+app_name+'/train.x')
    y_train=np.loadtxt('../data/'+app_name+'/train.y')
    x_test=np.loadtxt('../data/'+app_name+'/test.x')
    y_test=np.loadtxt('../data/'+app_name+'/test.y')
    return format_data(x_train),format_data(y_train),format_data(x_test),format_data(y_test)

def get_output_name(app_name, method, error_bound, iteration, epochA, epochC, net_A, net_C):
    output_name = '{}_{}_eb{}_it{}_epA{}_epC{}_netA{}_netC{}'.format(app_name,
                                                                     method,
                                                                     error_bound,
                                                                     iteration,
                                                                     epochA,
                                                                     epochC,
                                                                     '_'.join([str(x) for x in net_A]),
                                                                     '_'.join( [str(x) for x in net_C]))
    return output_name


# In[17]:

#Evaluation
def Error_of_Approximator(A,x,y): #Input one Approximator, x, y. Return  errorA[lenN]
    predictA=A.predict(x)
    errorA=[]
    for i in range(len(y)):
        errorA.append(error_measure(predictA[i],y[i]))
    return errorA
def Label_for_C(errorA,eb): #errorA[numA][lenN] return labelA[lenN][numA+1] for training C
    labelA=[]
    for i in range(lenN):
        labelA.append(numA)
        for j in range(numA):
            if (errorA[j][i]<=eb):
                labelA[i]=j
                break
    labelA=keras.utils.to_categorical(labelA,numA+1)
    return labelA

def evaluate(A, C, x_test, y_test, error_bound):
    lenN=len(x_test)
    #predictC
    predictC=C.predict(x_test)
    predictC=np.argmax(predictC,1)
    #predictA
    predictA=[]
    for i in range(numA):
        predictA.append(Error_of_Approximator(A[i],x_test,y_test))
    
    #calculate labelA Error Error a with C
    labelA=[]
    Er=[]
    Er_c=[]
    for i in range(lenN):
        Er.append(predictA[0][i])
        labelA.append(numA)
        min_error=error_bound
        for j in range(numA):
            if (predictA[j][i]<error_bound and min_error==error_bound):
                labelA[i]=j
                min_error=predictA[j][i]
            if (predictA[j][i]<Er[i]):
                Er[i]=predictA[j][i]
        if (predictC[i]<numA):
            Er_c.append(predictA[predictC[i]][i])
            
    #calculate the final results using predictA, labelA, Er, ErAwithC
    accuracy_of_C=sum([(labelA[i]==predictC[i]) for i in range(lenN)])/float(lenN)
    recall_of_C=sum([1.0 if labelA[i]<numA and predictC[i]<numA else 0 for i in range(lenN)]) / float(1e-10+sum([1 if (v<numA) else 0 for v in labelA]))
    invocation_of_C = float(sum([1 if (v<numA) else 0 for v in predictC])) / float(1e-10 + lenN)
    invocation_truly = float(sum([1 if (v<numA) else 0 for v in labelA])) / float(1e-10 + lenN)
    mean_relative_error_of_A = sum(Er) / float(1e-10 + len(Er))
    mean_relative_error_of_A_with_C = sum(Er_c) / (1e-10 + len(Er_c))
    
    return{'accuracy_of_C':accuracy_of_C,
           'recall_of_C':recall_of_C,
           'invocation_of_C':invocation_of_C,
           'invocation_truly':invocation_truly,
           'error_of_A_with_C':mean_relative_error_of_A_with_C,
           'error_of_A':mean_relative_error_of_A}


# In[5]:

#Accelerator & Classifier
def AcceleratorModel(net_list,type=0):
    if (len(net_list)<2):
        print('Error! input accelerator net structure is wrong!')
        exit(0)
    model=Sequential()
    model.add(Dense(net_list[1],input_shape=(net_list[0],)))
    model.add(Activation('sigmoid'))
    
    for i in net_list[2:]:
        model.add(Dense(i))
        model.add(Activation('sigmoid'))
        
    prop=[RMSprop(0.01),RMSprop(0.008),RMSprop(0.006),RMSprop(0.004),RMSprop(0.002)]
    #prop=[Adagrad(),Adagrad(),Adagrad()]
    model.compile(loss='mse',optimizer=prop[type],metrics=[metrics.mse])
    return model

def ClassifierModel(net_list):
    if (len(net_list)<2):
        print('Error! input classifier net structure is wrong!')
        exit(0)
    model=Sequential()
    model.add(Dense((net_list[1]),input_shape=(net_list[0],)))
    
    if (len(net_list)>2):
        model.add(Activation('sigmoid'))
        for i in net_list[2:-1]:
            model.add(Dense(i))
            model.add(Activation('sigmoid'))
        model.add(Dense(net_list[-1]))
        
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(0.01))
    return model


# In[29]:

#Training Block
def training(eb):
    global lenN
    #Load data
    x_train_origin,y_train_origin,x_test,y_test=load_data(app_name) #numpy type
    print ("Training data shape:",x_train_origin.shape,y_train_origin.shape)#Print the input shape, compare with the net_A and net_C, 
    print('Testing data shape:',x_test.shape,y_test.shape)                  #check whether they match each other
    lenN=len(x_train_origin)

    #Output file name
    outname=get_output_name(app_name, method, eb, iteration, epochA, epochC, net_A, net_C)
    f_results = open('../results/' + outname + '.csv', 'w')

    #Setting about approximator and classifier(There may be more than one Approximator, refer to acceleratorModel())
    A=[AcceleratorModel(net_A,i)for i in range(numA)]
    C=ClassifierModel(net_C)
    print("The Model A:")
    A[0].summary()
    print("The Model C:")
    C.summary()


    #每一个iteration�?
    #1.用上一轮的数据训练A（Using all data in train set in the 1st iteration�?
    #2.用所有Input做输入，predict A的output（这样能保证之前没有训练到的数据有加入训练数据的机会�?
    #3.A的输出和标准output比较，得出error，进而得出整个train数据集每一个数据能不能用A跑的label，用这个label训练C�?
    #4.让C对整个trainset进行预测，得到predictC。结合predictA和predictC产生下一个iteration的用于训练A的data�?算法�?
    #5.evaluate部分


    #Training data in the 1st iteration
    if (method=='MCMA_complementary'):
        print'complementary'
        x_train=[]
        y_train=[]
        x_train.append(x_train_origin)
        y_train.append(y_train_origin)
        A[0].fit(x_train[0],y_train[0],epochs=epochA,batch_size=batch_size,verbose=1)
        for i in range(1,numA):
            x_train.append([])
            y_train.append([])
            errorA=Error_of_Approximator(A[i-1],x_train[i-1],y_train[i-1])
            for j in range(len(errorA)):
                if (errorA[j]>eb):
                    x_train[i].append(x_train[i-1][j])
                    y_train[i].append(y_train[i-1][j])
            x_train[i]=np.asarray(x_train[i])
            y_train[i]=np.asarray(y_train[i])
            A[i].fit(x_train[i],y_train[i],epochs=epochA,batch_size=batch_size,verbose=1)
        print x_train[1].shape,y_train[1].shape
    else:
        x_train=[]
        y_train=[]
        for i in range(numA):
            x_train.append(x_train_origin)
            y_train.append(y_train_origin)
        x_train=np.asarray(x_train)
        y_train=np.asarray(y_train)

    for index in range(iteration):

        #Train approximators and get the prediction(the error from A in the whole training set)
        if (not(method=='MCMA_complementary' and index==0)):
            for i in range(numA):
                if (x_train[i].shape[0]>0):
                    A[i].fit(x_train[i],y_train[i],epochs=epochA,batch_size=batch_size,verbose=1) #Train A
        
        errorA=[]
        for i in range(numA):
            errorA.append(Error_of_Approximator(A[i],x_train_origin,y_train_origin))     
        labelA_trainC=Label_for_C(errorA,eb)

        #Use the label to train classifier
        C.fit(x_train_origin,labelA_trainC,epochs=epochC,batch_size=batch_size,verbose=1) #Using the label from A to train C
        predictC=C.predict(x_train_origin)
        predictC=np.argmax(predictC,1)

        #Get the training data in the next iterationx_train=[]
        x_train=[]
        y_train=[]
        for i in range(numA):
            x_train.append([])
            y_train.append([])
        for i in range(lenN):
            if (predictC[i]<numA):
                x_train[predictC[i]].append(x_train_origin[i])
                y_train[predictC[i]].append(y_train_origin[i])
        for i in range(numA):
            x_train[i]=np.array(x_train[i])
            y_train[i]=np.array(y_train[i])

        #Evaluation
        evaluate_output={'iteration':index}
        evaluate_output.update(evaluate(A,C,x_test,y_test,eb))
        if (index==0):
            keys=evaluate_output.keys()
            f_results.write(','.join(keys)+'\n')
        f_results.write(','.join([str(evaluate_output[v])for v in keys])+'\n')
        f_results.flush()
        print evaluate(A,C,x_test,y_test,eb)
    f_results.close()


# In[30]:

def main(app,meth,eA,eC):
    global app_name
    global method
    global epochA
    global epochC
    global numA
    global iteration
    global eb
    
    app_name=app
    method=meth
    
    numA=3
    if (method=='one_pass' or method=='iterative_training'):
        numA=1
    iteration=5
    if (method=='one_pass'):
        iteration=1
        
    epochA=eA
    epochC=eC
    get_net()
    print 'Using '+method+'. There are '+str(numA)+' approximators, train with '+str(iteration)+' iterations.'
    eb=np.loadtxt('../data/'+app_name+'_eb.csv')
    for i in range(len(eb)):
        training(eb[i])


# In[3]:

if __name__=="__main__":
    if (len(sys.argv)==5):
        #net_A = [int(x) for x in sys.argv[2].split('_')[1:]]
        #net_C = [int(x) for x in sys.argv[3].split('_')[1:]]
        #print(net_A)
        #print(net_C)
        main(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    else:
        print('Usage: python train_origin.py [benchmark_name] [method] [epochA] [epochC]')
        #print('#net_A|net_C: like a_6_8_8_1, c_6_8(the last layer depends on number of A)')
        exit(0)


# In[ ]:



