import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.optim as optim
import pandas as pd
import math
import time

import model # model.py


# set up matplotlib
isIpython = 'inline' in matplotlib.get_backend()
if isIpython:
    print("Ipython detected")
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") #DEBUG - force CPU


def plotRewards(showResult=False):
    #tickPlot = time.time()
    #tickPlotConfig = time.time()
    plt.figure(1)
    plt.clf()

    fig, axs = plt.subplots(2,2,num= 1)
    axs = axs.flatten()
    durationsT = torch.tensor(episodeRewards, dtype=torch.float)

    #tockPlotConfig = time.time()
    #print("Time taken to configure plot: ",tockPlotConfig-tickPlotConfig)

    #----------------------------------------------------------------------------------------------------------------
    # Plotting the reward values over time with 100 point moving average
    #tickAvg = time.time()
    # ax = axs[0]

    # if showResult:
    #     ax.set_title('Result')
    # else:
    #     ax.set_title('Training...')
    # ax.set_xlabel('Episode')
    # ax.set_ylabel('Reward')
    # plotDur = durationsT.numpy()
    # ax.plot(plotDur)
    # # Take 100 episode averages and plot them too
    # if len(durationsT) >= 100:
    #     means = durationsT.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means)).numpy()
    #     ax.plot(means)

    #plot the reward values over time with 100 point moving average
    #only plot last 10k points
    ax = axs[0]
    if showResult:
        ax.set_title('Result')
    else:
        ax.set_title('Training...')
    ax.set_xlabel('Last 1k Timesteps')
    ax.set_ylabel('Reward')
    plotDur = durationsT.numpy()
    ax.plot(plotDur[-1000:])
    # Take 100 episode averages and plot them too
    if len(durationsT) >= 100:
        means = durationsT.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means)).numpy()
        ax.plot(means[-1000:])
    

    #tockAvg = time.time()
    #print("Time taken to plot average: ",tockAvg-tickAvg)
    #----------------------------------------------------------------------------------------------------------------
    # Plot true values against predicted values
    #tickTruth = time.time()
    ax = axs[1]
    ax.set_title('True vs Predicted')
    ax.set_xlabel('Last 100 Timesteps')
    ax.set_ylabel('Value')
    ax.plot(chosenActions[-100:], label='Predicted')
    ax.plot(correctActions[-100:], label='True')

    #ax.plot(chosenActions, label='Predicted')
    #ax.plot(correctActions, label='True')
    ax.legend()

    #tockTruth = time.time()
    #print("Time taken to plot true vs predicted: ",tockTruth-tickTruth)
    #----------------------------------------------------------------------------------------------------------------
    #Plot epsilon values
    #tickEpsilon = time.time()
    ax = axs[2]
    ax.set_title('Epsilon')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Epsilon')
    ax.plot(epsValues)

    #tockEpsilon = time.time()
    #print("Time taken to plot epsilon: ",tockEpsilon-tickEpsilon)

    #----------------------------------------------------------------------------------------------------------------
    #Plot f1 score
    #tickF1 = time.time()
    ax = axs[3]
    ax.set_title('F1 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.plot(f1Scores,label='Train')
    ax.plot(f1TestScores,label='Test')
    ax.legend()
    #tockF1 = time.time()
    #print("Time taken to plot f1: ",tockF1-tickF1)

    #tickPause = time.time()
    plt.pause(0.001)
    if isIpython:
        if not showResult:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    #tockPause = time.time()
    #print("Time taken to pause: ",tockPause-tickPause)


    #tockPlot = time.time()
    #print("Time taken to plot: ",tockPlot-tickPlot)


def rewarding(action,iteration,labels):
    if action == labels[iteration].item(): #if correct action
      if action == 0: #if no anomaly
         return 2
      else: #if anomaly
         return 5
    else: #if wrong action
      if action == 0: #says no anomaly but there is
         return -1
      else: #says there is anomaly but there isn't
         return -5


# BATCHSIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPSSTART is the starting value of epsilon
# EPSEND is the final value of epsilon
# EPSDECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimiser

#tickSetup = time.time()

BATCHSIZE = 128
GAMMA = 0.5
EPSSTART = 0.8
EPSEND = 0.075
EPSDECAY = 2000
TAU = 0.001
LR = 1e-4


nActions = 2 # 0th -> no anomaly, 1st -> anomaly

TAG = pd.read_csv("mergedKDDTrain.csv",header=0)
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


#repeat to load test data
TAGTest = pd.read_csv("mergedKDDTest.csv",header=0)
TAGTest,outcomeTest = TAGTest.iloc[:,1:-1],TAGTest.iloc[:,len(TAGTest.keys())-1] # split into observations and outcomes
namesTest = TAGTest.iloc[0].index.values
TAGTest = torch.tensor(TAGTest.values, dtype=torch.float32)

seqTest = []
labelsTest = []
for i in range(0,len(TAGTest)-steps+1): 
    seqTest.append(TAGTest[i:i+steps])
    labelsTest.append(outcomeTest.iloc[i+steps-1]) # outcome is the last observation in the sequence
dataTest = torch.stack(seqTest)
labelsTest = torch.tensor(labelsTest)

datasetTest = torch.utils.data.TensorDataset(dataTest,labelsTest)
dataLoaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=BATCHSIZE)


nObservations = data.shape[2]


policyNet = model.DQN(nObservations, nActions).to(device)
targetNet = model.DQN(nObservations, nActions).to(device)
targetNet.load_state_dict(policyNet.state_dict())

optimiser = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)
memory = model.ReplayMemory(300)

torch.save(targetNet.state_dict(), "targetNet.pth")

stepsDone = 0

episodeRewards = []
chosenActions = []
correctActions = []
epsValues = []
f1Scores =[]
f1TestScores = []




epoch = 1500 #Run through all the data 'epoch' times

#tockSetup = time.time()
#print("Time taken to setup: ",tockSetup-tickSetup)

counter = 0
highestF1 = 0
savedModels = 0
for eachEpoch in range(epoch):
    thisEpochActions = []
    thisEpochCorrect = []
    #tickEpochS = time.time()
    #print("Epoch: ",eachEpoch)
    #tickEpochBatch = time.time()
    for iEpisode, (batchedState,batchedOutcome) in enumerate(dataLoader):
        counter += 1
        #tickEpisode = time.time()

        # Might be uneccessary
        batchedState = batchedState.to(device)
        batchedOutcome = batchedOutcome.to(device)

        actions,stepsDone = model.selectAction(batchedState, policyNet, device, stepsDone, EPSSTART, EPSEND, EPSDECAY)
        rewards = torch.tensor([rewarding(action.item(),index,batchedOutcome) for index,action in enumerate(actions)], device=device)
        
        if len(batchedState) > 1:
            nextStates = batchedState[1:]
        else:
            nextStates = torch.zeros_like(batchedState)  # Padding for the last state if needed
        
        for state,action,nextState,reward,correctAction in zip(batchedState,actions,nextStates,rewards,batchedOutcome):
            memory.push(state,action,nextState,reward,correctAction)
      
        model.optimiseModel(memory,BATCHSIZE,GAMMA,policyNet,targetNet,optimiser,device)

        targetNetStateDict = targetNet.state_dict()
        policyNetStateDict = policyNet.state_dict()
        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key]*TAU + targetNetStateDict[key]*(1-TAU)
        targetNet.load_state_dict(targetNetStateDict)

        thisEpochActions.extend(actions.tolist())
        thisEpochCorrect.extend(batchedOutcome.tolist())

        episodeRewards.extend(rewards.tolist())
        chosenActions.extend(actions.tolist())
        correctActions.extend(batchedOutcome.tolist())


        epsValues.append(EPSEND + (EPSSTART - EPSEND) * math.exp(-1. * stepsDone / EPSDECAY))
        #tockEpisode = time.time()
        #print("Time taken for episode: ",tockEpisode-tickEpisode)
        if counter % 100 == 0:
            plotRewards()

    #tockEpochBatch = time.time()
    #print("Time taken for batch: ",tockEpochBatch-tickEpochBatch)

    #tickEpoch = time.time()
    # Convert lists to tensors
    thisEpochActionsTensor = torch.tensor(thisEpochActions)
    thisEpochCorrectTensor = torch.tensor(thisEpochCorrect)

    # Identify changes in the anomaly detection state
    diff = thisEpochCorrectTensor.diff(prepend=torch.tensor([0]))
    starts = (diff == 1).nonzero(as_tuple=True)[0]
    ends = (diff == -1).nonzero(as_tuple=True)[0]

    # If there's an open anomaly at the end, consider it closed
    if len(starts) > len(ends):
        ends = torch.cat((ends, torch.tensor([len(thisEpochCorrectTensor)])))

    # Count detections within each anomaly window
    detections = torch.zeros_like(starts, dtype=torch.bool)
    for i, (start, end) in enumerate(zip(starts, ends)):
        detections[i] = (thisEpochActionsTensor[start:end] == 1).any()

    # Calculate TP and FN
    TP = detections.sum().item()
    FN = len(detections) - TP

    # Calculate FP and TN
    normal_indices = (thisEpochCorrectTensor == 0).nonzero(as_tuple=True)[0]
    FP = (thisEpochActionsTensor[normal_indices] == 1).sum().item()
    TN = len(normal_indices) - FP

    precision = TP/(TP+FP+1e-10) #avoid division by zero
    recall = TP/(TP+FN+1e-10) #avoid division by zero
    f1 = 2*(precision*recall)/(precision+recall+1e-10) #avoid division by zero
    f1Scores.append(f1)

    #tockEpoch = time.time()
    #print("Time taken for f1: ",tockEpoch-tickEpoch)
    #tickEpoch = time.time()

    thisEpochActions = []
    thisEpochCorrect = []
        #benchmark model against test set
    with torch.no_grad():
        
        for iEpisode, (batchedState,batchedOutcome) in enumerate(dataLoaderTest):
            batchedState = batchedState.to(device)
            batchedOutcome = batchedOutcome.to(device)

            qVals = targetNet(batchedState)
            actions = qVals.max(1)[1].view(-1, 1)
            thisEpochActions.extend(actions.tolist())
            thisEpochCorrect.extend(batchedOutcome.tolist())

    thisEpochActionsTensorTest = torch.tensor(thisEpochActions)
    thisEpochCorrectTensorTest = torch.tensor(thisEpochCorrect)

    # Identify changes in the anomaly detection state for the test set
    diffTest = thisEpochCorrectTensorTest.diff(prepend=torch.tensor([0]))
    startsTest = (diffTest == 1).nonzero(as_tuple=True)[0]
    endsTest = (diffTest == -1).nonzero(as_tuple=True)[0]

    if len(startsTest) > len(endsTest):
        endsTest = torch.cat((endsTest, torch.tensor([len(thisEpochCorrectTensorTest)])))

    detectionsTest = torch.zeros_like(startsTest, dtype=torch.bool)
    for i, (start, end) in enumerate(zip(startsTest, endsTest)):
        detectionsTest[i] = (thisEpochActionsTensorTest[start:end] == 1).any()

    TPt = detectionsTest.sum().item()
    FNt = len(detectionsTest) - TPt

    normal_indices_test = (thisEpochCorrectTensorTest == 0).nonzero(as_tuple=True)[0]
    FPt = (thisEpochActionsTensorTest[normal_indices_test] == 1).sum().item()
    TNt = len(normal_indices_test) - FPt

    precisionTest = TPt / (TPt + FPt + 1e-10)  # avoid division by zero
    recallTest = TPt / (TPt + FNt + 1e-10)  # avoid division by zero
    f1Test = 2 * (precisionTest * recallTest) / (precisionTest + recallTest + 1e-10)  # avoid division by zero
    f1TestScores.append(f1Test)

    #tockEpoch = time.time()
    #print("Time taken for f1 test: ",tockEpoch-tickEpoch)

    #tickplot = time.time()

    #tockplot = time.time()
    #print("Time taken to plot: ",tockplot-tickplot)

    #tickEpoch = time.time()
    if f1 > highestF1:
        highestF1 = f1
        #save model
        savedModels += 1
        torch.save(targetNet.state_dict(), f"targetNet{savedModels}.pth")
        with open("highestF1.txt","w") as f:
            f.write(str(highestF1))
    
    #tockEpoch = time.time()
    #print("Time taken for save: ",tockEpoch-tickEpoch)
    #tockEpochE = time.time()
    #print("Time taken for epoch: ",tockEpochE-tickEpochS)


       
print('Complete')
plotRewards(showResult=True)
plt.ioff()
plt.show()