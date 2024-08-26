import pandas as pd

test = pd.read_csv("KDDTest.csv",header=0)
test.to_csv("KDDTest2.csv",index=True)

