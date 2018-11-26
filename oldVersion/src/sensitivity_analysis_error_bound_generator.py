
# coding: utf-8

# In[2]:

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


# Parametres
# ====

# In[3]:

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


# In[4]:

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


# Input data processing
# ======

# In[5]:

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


# Output evaludation processing
# =========

# In[6]:

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


# Evaluation
# ======

# In[7]:

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


# In[8]:

def evaluate(A, C, x_test, y_test, error_bound):
    lenN=x_test
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
                Er[i]=predictA[j][i][0]
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


# Accelerator & Classifier
# ===

# In[9]:

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


# In[10]:

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


# Error bound Generation
# ========

# In[11]:

def generate_eb():
    global A
    #Setting of parameters
    get_net()
    global eb
    global lenN
    
    #Reading the dataset
    x_train_origin,y_train_origin,x_test,y_test=load_data(app_name) #numpy type
    lenN=len(x_train_origin)
    print ("Training data shape:",x_train_origin.shape,y_train_origin.shape)#Print the input shape, compare with the net_A and net_C, 
    print('Testing data shape:',x_test.shape,y_test.shape)                  #check whether they match each other
    
    #Setting the neural network
    A=[AcceleratorModel(net_A,i)for i in range(numA)]
    print("The Model A:")
    A[0].summary()
    
    #Training the approximator
    A[0].fit(x_train_origin,y_train_origin,epochs=epochA,batch_size=batch_size,verbose=1)
    
    #Generate error_bound(30% 50% 80%)
    errorA=Error_of_Approximator(A[0],x_train_origin,y_train_origin)
    sortError=sorted(errorA)
    eb=[]
    for i in range(1,10):
        eb.append(sortError[int(lenN*0.1*i)])
    print eb
    f_results = open('../data/' + app_name+ '_eb.csv', 'w')
    f_results.write(' '.join([str(eb[i]) for i in range(len(eb))]))
    f_results.close()    


# In[12]:

def main(app,eA,eC):
    global app_name
    global epochA
    global epochC
    app_name=app
    epochA=eA
    epochC=eC
    
    generate_eb()


# In[ ]:

if __name__=="__main__":
    if (len(sys.argv)==4):
        #net_A = [int(x) for x in sys.argv[2].split('_')[1:]]
        #net_C = [int(x) for x in sys.argv[3].split('_')[1:]]
        #print(net_A)
        #print(net_C)
        main(sys.argv[1], int(sys.argv[2]),int(sys.argv[3]))
    else:
        print('Usage: python train_origin.py [benchmark_name] [epochA] [epochC]')
        #print('#net_A|net_C: like a_6_8_8_1, c_6_8(the last layer depends on number of A)')
        exit(0)

