import pandas as pd

TAG = pd.read_csv("merged.csv",header=0)
timestamp,df,outcome = TAG.iloc[:,0],TAG.iloc[:,1:7],TAG.iloc[:,7] # split into observations and outcomes 

# Ignore the clear data for now

for i in range(len(df.columns)):
    dfBaseMin = df.iloc[:,i].min()
    dfBaseMax = df.iloc[:,i].max()
    rangeBase = dfBaseMax - dfBaseMin
    df.iloc[:,i] = (df.iloc[:,i] - dfBaseMin) / rangeBase

TAG = pd.concat([timestamp,df,outcome],axis=1)
TAG.to_csv("normalised2.csv",index=False)