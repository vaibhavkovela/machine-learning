import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test_train_split(a,s,dat):   #split a is split % , s is length of dataset , dat is dataset
    import random 
    import math   
    a=math.floor((s/100)*a)
    trn=list()
    tst=list()
    testi=[]
    testx,trainx,trainy,testy=[],[],[],[]
    for i in range(a):
        random.randrange(s)
        tst.append(dat.iloc[i])
        testi.append(i)
    for i in range(s):
        if i not in testi:
            trn.append(dat.iloc[i])
    testx=tst[['FUELCONSUMPTION_COMB','ENGINESIZE']]
    testx['One']=[1 for i in range(len(tst))]
    testy=tst[['CO2EMISSIONS']]
    trainx=trn[['FUELCONSUMPTION_COMB','ENGINESIZE']]
    trainx['One']=[1 for i in range(len(trn))]
    trainy=trn[['CO2EMISSIONS']]
    cols=testx.columns.tolist()
    cols=cols[-1:]+cols[:-1]
    trainx=trainx[cols]
    testx=testx[cols]
    return testx,trainx,testy,trainy 

def CostFunct(t,x,y):
    tot=(np.subtract(np.dot(trainx,t),y)**2)
    tot=sum(tot)
    tot/=(2*len(x))
    return tot

def GradDesc(x,y,tsx,tsy,alpha=0.00000001):
    tl=list()
    #lcst=8000
    result=[]
    t=[[0],[0],[0]]
    a=list()
    for i in range(15):
        E=np.subtract(np.dot(x,t),y)
        X=np.dot(np.transpose(E),x)
        tmp=(np.dot((alpha/len(x)),np.transpose(X)))
        t=np.subtract(t,tmp)
        cst=CostFunct(t,x,y)
        a.append(cst.tolist())
        # if(cst<lcst):
        #     lt=t
        #     lcst=cst
        tl.append(t.tolist())
        result.append(predict(t,tsx,tsy))
    oneton=[i for i in range(15)]
    from mpl_toolkits.mplot3d import Axes3D
    fig1 = plt.figure(figsize=(30,10))
    ax=fig1.add_axes([0,0,1,1])
    ax.scatter(oneton, result, color='r')
    ax.set_xlabel('No of Iterations')
    ax.set_ylabel('Error')
    ax.set_title('scatter plot')
    plt.show()
    return t

def predict(thet,testx,testy):  
    testx=np.array(testx)
    res=testx.dot(thet)
    return mse(testy,res)

def mse(a,b):
    a=np.array(a)
    b=np.array(b)
    c=np.square(a-b)
    c=np.sum(c)
    c=c/len(a)
    return c

def least(a):
    a=np.sort(a)
    return a[0]

if __name__ == '__main__':
    filename='filename'
    f=pd.read_csv(filename)   #import the data frame
    keys=f.columns
    a=f[['FUELCONSUMPTION_COMB','ENGINESIZE','CO2EMISSIONS']] # select data attributes on your choice 
    x,y,X,Y=test_train_split(25,len(f),a)
    t=GradDesc(x.values.tolist(),y.values.tolist(),X.values.tolist(),Y.values.tolist())
    t=least(t)
    print(t)
