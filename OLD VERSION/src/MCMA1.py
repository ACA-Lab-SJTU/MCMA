
# coding: utf-8

# In[32]:

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

# In[34]:

app_name='bessel_Jnu'
net_A=[2,4,4,1]
numA=3
net_C=[2,4,4,numA+1]
iteration=2
error_bound=0.1



epochA=20
epochC=20
batch_size=128

error_type="absolute_error"

def error_compare(error_type):
    if (error_type=="absolute_error"):
        return error.absolute_error
    else:
        return error.relative_error
error_measure=error_compare(error_type)


# Input data processing
# ======

# In[35]:

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

# In[36]:

def get_output_name(app_name, error_bound,epochA, epochC,net_A, net_C):
    def get_name(iteration):
        output_name = '{}_it{}_eb{}_epA{}_epC{}_netA{}_netC{}'.format(app_name,
                                                                      iteration,
                                                                      error_bound,
                                                                      epochA, epochC,
                                                                      '_'.join(
                                                                          [str(x) for x in net_A]),
                                                                      '_'.join(
                                                                          [str(x) for x in net_C]))
        return output_name
    return get_name


# Evaluation
# ======

# In[37]:

def evaluate(A,C,x_test,y_test):
    N=len(x_test)
    
    predictC=C.predict(x_test)
    predictC=np.argmax(predictC,1)
    
    predictA=[]
    for i in range(numA):
        predictA.append(A[i].predict(x_test))
    for i in range(numA):
        for j in range(N):
            predictA[i][j]=error_measure(predictA[i][j],y_test[j])
            
    labelA=[]
    Er=[]
    Er_c=[]
    for i in range(N):
        Er.append(predictA[0][i][0])
        labelA.append(numA)
        min_error=error_bound
        for j in range(numA):
            if (predictA[j][i][0]<min_error):
                labelA[i]=j
                min_error=predictA[j][i][0]
            if (predictA[j][i][0]<Er[i]):
                Er[i]=predictA[j][i][0]
        if (predictC[i]<numA):
            Er_c.append(predictA[predictC[i]][i][0])

    accuracy_of_C=sum([(labelA[i]==predictC[i]) for i in range(N)])/float(N)
    recall_of_C=sum([1.0 if labelA[i]<numA and predictC[i]<numA else 0 for i in range(N)]) / float(sum([1 if (v<numA) else 0 for v in labelA]))
    invocation_of_C = float(sum([1 if (v<numA) else 0 for v in predictC])) / float(1e-10 + N)
    invocation_truly = float(sum([1 if (v<numA) else 0 for v in labelA])) / float(1e-10 + N)

    mean_relative_error_of_A = sum(Er) / float(1e-10 + len(Er))
    mean_relative_error_of_A_with_C = sum(Er_c) / (1e-10 + len(Er_c))
    return{'accuracy_of_C':accuracy_of_C,
           'recall_of_C':recall_of_C,
           'invocation_of_C':invocation_of_C,
           'invocation_truly':invocation_truly,
           'error_of_A_with_C':mean_relative_error_of_A_with_C}


# Accelerator & Classifier
# ===

# In[39]:

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


# In[40]:

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


# Training Block
# ====

# In[45]:

def Training(app_name,net_A,net_C,iteration,error_bound,epochA,epochC,batch_size):
    #Load data
    x_train_origin,y_train_origin,x_test,y_test=load_data(app_name) #numpy type
    print ("Training data shape:",x_train_origin.shape,y_train_origin.shape)#Print the input shape, compare with the net_A and net_C, 
    print('Testing data shape:',x_test.shape,y_test.shape)                  #check whether they match each other

    #Output file name
    outname=get_output_name(app_name, error_bound,epochA, epochC,net_A, net_C)
    f_results = open('../results/' + outname(iteration) + '.csv', 'w')

    #Setting about approximator and classifier(There may be more than one Approximator, refer to acceleratorModel())
    A=[AcceleratorModel(net_A,i)for i in range(numA)]
    C=ClassifierModel(net_C)
    print("The Model A:")
    A[0].summary()
    print("The Model C:")
    C.summary()
    lenN=len(x_train_origin)

    #每一个iteration�?
    #1.用上一轮的数据训练A（Using all data in train set in the 1st iteration�?
    #2.用所有Input做输入，predict A的output（这样能保证之前没有训练到的数据有加入训练数据的机会�?
    #3.A的输出和标准output比较，得出error，进而得出整个train数据集每一个数据能不能用A跑的label，用这个label训练C�?
    #4.让C对整个trainset进行预测，得到predictC。结合predictA和predictC产生下一个iteration的用于训练A的data�?算法�?
    #5.evaluate部分


    #Training data in the 1st iteration
    x_train=[]
    y_train=[]
    for i in range(numA):
        x_train.append(x_train_origin)
        y_train.append(y_train_origin)
    x_train=np.asarray(x_train)
    y_train=np.asarray(y_train)

    for index in range(iteration):
        #Train approximators and get the prediction(the error from A in the whole training set)
        predictA=[]
        for i in range(numA):
            A[i].fit(x_train[i],y_train[i],epochs=epochA,batch_size=batch_size,verbose=1) #Train A
            predictA.append(A[i].predict(x_train_origin)) #Predict for A on x_train
            for j in range(lenN):
                predictA[i][j]=error_measure(predictA[i][j],y_train_origin[j])
        #From the prediction above, determine the label      
        labelA=[]
        for i in range(lenN):
            labelA.append(numA)
            for j in range(numA):
                if (predictA[j][i][0]<error_bound):
                    labelA[i]=j
                    break
        labelA_trainC=keras.utils.to_categorical(labelA,numA+1) #change to the type for training C

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
        evaluate_output.update(evaluate(A,C,x_test,y_test))
        if (index==0):
            keys=evaluate_output.keys()
            f_results.write(','.join(keys)+'\n')
        f_results.write(','.join([str(evaluate_output[v])for v in keys])+'\n')
        f_results.flush()
        print evaluate(A,C,x_test,y_test)
    f_results.close()


# In[ ]:

def main(app_name,net_A,net_C,iteration,error_bound,epochA,epochC,batch_size):
    Training(app_name,net_A,net_C,iteration,error_bound,epochA,epochC,batch_size)


# In[46]:

if __name__=="__main__":
    if (len(sys.argv)==9):
        net_A = [int(x) for x in sys.argv[2].split('_')[1:]]
        net_C = [int(x) for x in sys.argv[3].split('_')[1:]]
        print(net_A)
        print(net_C)
        main(sys.argv[1], net_A, net_C, int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]),
             int(sys.argv[7]), int(sys.argv[8]))
    else:
        print('Usage: python train_origin.py [benchmark_name] [net_A] [net_C] [iteration] [error_bound] [epochA] [epochC] [batch_size]')
        print('#net_A|net_C: like a_6_8_8_1, c_6_8(the last layer depends on number of A)')
        exit(0)


# In[47]:




# In[48]:




# In[49]:




# In[ ]:



