import torch
import model
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") #DEBUG - force CPU

BATCHSIZE = 128
GAMMA = 0.5
EPSSTART = 0.8
EPSEND = 0.075
EPSDECAY = 2000
TAU = 0.001
LR = 1e-3


nActions = 2 # 0th -> no anomaly, 1st -> anomaly

TAG = pd.read_csv("mergedSynth.csv",header=0)
TAG,outcome = TAG.iloc[:,1:-1],TAG.iloc[:,len(TAG.keys())-1] # split into observations and outcomes
names = TAG.iloc[0].index.values

#Tag to a tensor
TAG = torch.tensor(TAG.values, dtype=torch.float32)

# Each episode is a sequence of observations with a single outcomes - 1 if at least one of the observations is 1, 0 otherwise

#Create a sliding window of 'steps' time steps
steps = 10  # 10 points per episode
seq = []
labels = []
for i in range(0,len(TAG)-steps+1): 
    seq.append(TAG[i:i+steps])
    labels.append(outcome.iloc[i+steps-1]) # outcome is the last observation in the sequence
data = torch.stack(seq)
labels = torch.tensor(labels)


dataset = torch.utils.data.TensorDataset(data,labels)
dataLoader = torch.utils.data.DataLoader(dataset,batch_size=BATCHSIZE)
#state is the observation of the environment
nObservations = data.shape[1]

policyNet = model.DQN(nObservations, nActions).to(device)
targetNet = model.DQN(nObservations, nActions).to(device)
targetNet.load_state_dict(policyNet.state_dict())

targetNet.load_state_dict(torch.load("targetNet.pth"))
targetNet.eval()

network = targetNet

# Run model on test data
normalQs, anomalyQs = [], []
with torch.no_grad():
    for iEpisode, (batchedState, batchedOutcome) in enumerate(dataLoader):
        batchState = batchedState.to(device)
        qValues = network(batchedState)
        
        # Assuming qValues output shape is [batch_size, seq_len, nActions]
        normalQs.extend(qValues[:, 0].cpu().numpy())
        anomalyQs.extend(qValues[:, 1].cpu().numpy())

# Convert lists to numpy arrays
normalQs = np.array(normalQs)
anomalyQs = np.array(anomalyQs)
score = anomalyQs - normalQs  # Compute the score for ROC

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(labels, score)
aucroc = auc(fpr, tpr)
print("AUC ROC: ", aucroc)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % aucroc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
