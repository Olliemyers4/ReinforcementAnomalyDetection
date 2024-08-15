import pandas as pd

normal = pd.read_csv('KDDTrainNumerical.csv')
recon = pd.read_csv('KDDAE.csv')

merged = pd.merge(normal, recon, on='timestamp', how='inner')
 
anom = merged.pop('class')
merged.insert(len(merged.columns), 'class', anom)
merged.to_csv('mergedKDD.csv', index=False)