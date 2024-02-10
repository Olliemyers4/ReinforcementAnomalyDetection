import os


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
            csvFiles.append(os.path.join(root, file))

print("Found the following csv files:")
for i in range(0,len(csvFiles)):
    print(f"File {i}: {file}")

print("Which file would you like to use?")
fileIndex = int(input("File Index:"))
file = csvFiles[fileIndex]

with open(file, "r") as f:
    lines = f.readlines()

print("The first 5 lines of the file are:")
for i in range(0,5):
    print(lines[i])