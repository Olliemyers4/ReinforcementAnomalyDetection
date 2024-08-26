import pandas as pd

normal = pd.read_csv('KDDTestNumerical.csv')
recon = pd.read_csv('KDDTestRecon.csv')

merged = pd.merge(normal, recon, on='timestamp', how='inner')
 
anom = merged.pop('class')
merged.insert(len(merged.columns), 'class', anom)
merged.to_csv('mergedKDDTest.csv', index=False)