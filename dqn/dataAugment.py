import pandas as pd

normal = pd.read_csv('testData.csv')
recon = pd.read_csv('recon.csv')

merged = pd.merge(normal, recon, on='timestamp', how='inner')
 
anom = merged.pop('is_anomaly')
merged.insert(len(merged.columns), 'is_anomaly', anom)
merged.to_csv('merged.csv', index=False)