import numpy as np
import pandas as pd
import matplotlib as plt

filename='Your_file_name'

def mean_normalise(df):
    """
    This function is used to normalise the
    Dataset using each attributes mean """
    for x in df.keys[:-1]:
        df[x] = [i / df[x].mean() for i in df[x]]
    return df


def splitnselect(dat, perc):
    """
    this function splits gives us chance to  attributes
    selection we either need to mention the number or the
    change some code and send some additional lists to split the code exclusively
    with your need and moreover  you need to pass the percentage of test split for 
    splitting the dataset """
    s = len(dat)
    perc = (int)((s / 100) * perc)
    li = np.random.permutation(s)
    trnx = dat.iloc[li[:perc], :-1]
    trny = dat.iloc[li[:perc], [-1]]
    tstx = dat.iloc[li[perc:], :-1]
    tsty = dat.iloc[li[perc:], [-1]]
    trnx = np.array(trnx)
    trny = np.array(trny)
    tstx = np.array(tstx)
    tsty = np.array(tsty)
    return trnx, trny, tstx, tsty


def log(a): #log Function
    return np.log(a)


def h(x, t):   #hyposthesis function (sigmoid)
    x = np.array(x)
    t = np.array(t)
    a = np.dot(x, t)
    a = 1 / (1 + np.exp(-a))
    return a


def CostFunct(t, x, y):   #cost function for sigmoid function
    tot = np.add((np.multiply(y, log(h(x, t)))), (np.multiply(1 - y, log(h(x, t)))))
    tot = sum(tot)
    tot /= -1 * len(x)
    return tot


def GradDesc(tx, ty, Tx, Ty, x, epch=80, alpha=5e-3):
    """
    tx : train input 
    ty : train output
    Tx : test input 
    Ty : test output 
    epch : epoch count
    alpha : step size
    This is a Epoch based implementation of Batch Gradient Descent Algorithm """
    t = np.zeros((x, 1))
    for i in range(epch):
        tmp = np.dot(tx.T, (np.subtract(h(tx, t), ty)))
        tmp = (alpha / len(tx)) * (tmp)
        t = t - tmp
        cst = CostFunct(t, tx, ty)
    return cst, t


def Accuracy(Tx, Ty, thet):
    a = h(Tx, thet)
    result = a > 0.5
    result = [int(i) for i in result]
    result = sum(result)
    result /= len(Tx)
    print("resulting accuracy", result * 100)


if __name__ == '__main__':
    df = pd.read_csv(filaneme)
    df = mean_normalise(df)
    df.insert(0, "bias", [1 for i in range(len(df))])
    df = mean_normalise(df)
    tx, ty, Tx, Ty = splitnselect(df, 33)
    a, b = GradDesc(Tx, Ty, tx, ty, 14, 10)
    print("theta ; ", b)
    print("Cost : ", a)
    Accuracy(Tx, Ty, b)
