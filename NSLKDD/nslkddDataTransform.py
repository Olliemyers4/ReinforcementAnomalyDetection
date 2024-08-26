#we want to remove categorical features from the dataset by converting them to numerical values
#we want to change outcome to 1 for anomaly and 0 for normal

import pandas as pd
import numpy as np

df = pd.read_csv("KDDTest2.csv",header=0)



#convert categorical features to numerical values
df["protocol_type"] = df["protocol_type"].astype("category").cat.codes
df["service"] = df["service"].astype("category").cat.codes
df["flag"] = df["flag"].astype("category").cat.codes
df["class"] = df["class"].astype("category").cat.codes

df.to_csv("KDDTestNumerical.csv",index=False)


#remove rows where anomaly is 1
df = df[df["class"] != 1]
df.to_csv("KDDTestAE.csv",index=False)