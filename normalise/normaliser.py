import pandas as pd

TAG = pd.read_csv("merged.csv",header=0)
timestamp,df,recon,outcome = TAG.iloc[:,0],TAG.iloc[:,1:4],TAG.iloc[:,4:7],TAG.iloc[:,7] # split into observations and outcomes also remove all the reconstruction error cols

base = pd.read_csv("testClear.csv",header=0)
timestampBase,dfBase,outcomeBase = base.iloc[:,0],base.iloc[:,1:4],base.iloc[:,4]

for i in range(len(dfBase.columns)):
    dfBaseMin = dfBase.iloc[:,i].min()
    dfBaseMax = dfBase.iloc[:,i].max()
    rangeBase = dfBaseMax - dfBaseMin
    df.iloc[:,i] = (df.iloc[:,i] - dfBaseMin) / rangeBase

TAG = pd.concat([timestamp,df,recon,outcome],axis=1)
TAG.to_csv("normalised.csv",index=False)