import os
import pandas as pd
from matplotlib import pyplot as plt

cwd = os.getcwd()
topFolderForGenerated = os.path.join(cwd, "generated-timeseries")
# File structure of this folder is as follows:
# generated-timeseries
# ├── folder1
# │   ├── generated1.csv
# ├── foldern
# │   ├── generatedn.csv
# ├── overview.yaml # This file contains the overview of the generated timeseries (that was most recently generated)
# Only care about these csv files

# get all csv files
csvFiles = []
for root, dirs, files in os.walk(topFolderForGenerated):
    for file in files:
        if file.endswith(".csv"):
            csvFiles.append(os.path.join(root,file))


print("Found the following csv files:")
for i in range(0,len(csvFiles)):
    #we also need the folder name because all the csv files are named the same - test.csv
    print(f"File {i}: {csvFiles[i]}")

print("Which file would you like to use?")
fileIndex = int(input("File Index:"))
file = csvFiles[fileIndex]

df = pd.read_csv(file)

timeseries = len(df.columns) - 2 # -2 for timestamp is_anomaly
#As long as we dont do any weird shuffling we can ignore the timestamp column and anomaly column
#ignore 0th col and take last col for each col in the timeseries
fig, axs = plt.subplots(timeseries+1)
for i in range(1, timeseries+1):
    axs[i-1].plot(df[df.columns[i]])
    axs[i-1].set_title(df.columns[i])
#Add the anomaly
axs[timeseries].plot(df[df.columns[timeseries+1]])
axs[timeseries].set_title(df.columns[timeseries+1])
plt.show(block=False) #we can continue getting UserInput

totalTimeSeries = int(input("How many total time series do you want?"))

newDF = pd.DataFrame()
newDF.insert(0, "timestamp",df[df.columns[0]],True) #timestamp

for i in range(0,totalTimeSeries):
    print(f"Timeseries {i}:")
    print("\n")
    option = int(input("0 for using an existing time series, 1 for a linear combination:"))

    if option == 0:
        waveIndex = int(input("Which timeseries would you like to use?"))
        newDF.insert(len(newDF.columns), f"value-{i}",df[df.columns[waveIndex]],True)
    else:
        combined = int(input("How many timeseries would you like to combine?"))
        combinedSeries = [0] *len(df[df.columns[0]])
        for j in range(0,combined):
            waveIndex = int(input("Which timeseries would you like to use?"))
            multiplier = float(input("What multiplier would you like to use?(float)"))
            multiplied = df[df.columns[waveIndex]]*multiplier
            combinedSeries = [x + y for x, y in zip(combinedSeries, multiplied)]
        newDF.insert(len(newDF.columns), f"value-{i}",combinedSeries,True)

newDF.insert(len(newDF.columns), "is_anomaly",df[df.columns[timeseries+1]],True)
newDF.to_csv(os.path.join(cwd, "combined.csv"), index=False)
print("File saved as combined.csv")