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
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") #DEBUG - force CPU


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
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plotDur = durationsT.numpy()
    ax.plot(plotDur[-10000:])
    # Take 100 episode averages and plot them too
    if len(durationsT) >= 100:
        means = durationsT.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means)).numpy()
        ax.plot(means[-10000:])
    

    #tockAvg = time.time()
    #print("Time taken to plot average: ",tockAvg-tickAvg)
    #----------------------------------------------------------------------------------------------------------------
    # Plot true values against predicted values
    tickTruth = time.time()
    ax = axs[1]
    ax.set_title('True vs Predicted')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')

    ax.plot(chosenActions, label='Predicted')
    ax.plot(correctActions, label='True')
    ax.legend()

    #tockTruth = time.time()
    #print("Time taken to plot true vs predicted: ",tockTruth-tickTruth)
    #----------------------------------------------------------------------------------------------------------------
    #Plot epsilon values
    #tickEpsilon = time.time()
    ax = axs[2]
    ax.set_title('Epsilon')
    ax.set_xlabel('Timestep')
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
    ax.plot(f1Scores)
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
steps = 50  # 10 points per episode
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

optimiser = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)
memory = model.ReplayMemory(300)

torch.save(targetNet.state_dict(), "targetNet.pth")

stepsDone = 0

episodeRewards = []
chosenActions = []
correctActions = []
epsValues = []
f1Scores =[]




epoch = 1500 #Run through all the data 'epoch' times

#tockSetup = time.time()
#print("Time taken to setup: ",tockSetup-tickSetup)

counter = 0
highestF1 = 0
savedModels = 0
for eachEpoch in range(epoch):
    thisEpochActions = []
    thisEpochCorrect = []
    #tickEpoch = time.time()
    #print("Epoch: ",eachEpoch)
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

    #At the end of the epoch handle the accuracy etc
    anomDetected = 0
    anomWindow = False

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(thisEpochCorrect)):
        if thisEpochCorrect[i] == 1:
            anomWindow = True
            if thisEpochActions[i][0] == 1:
                anomDetected = 1
        else:
            if anomWindow:
                anomWindow = False
                if anomDetected: #Anomaly detected in the window
                    TP += 1
                else:
                    FN += 1
                anomDetected = 0
            if thisEpochActions[i][0] == 1:
                FP += 1
            else:
                TN += 1

    precision = TP/(TP+FP+1e-10) #avoid division by zero
    recall = TP/(TP+FN+1e-10) #avoid division by zero
    f1 = 2*(precision*recall)/(precision+recall+1e-10) #avoid division by zero
    f1Scores.append(f1)
    if f1 > highestF1:
        highestF1 = f1
        #save model
        savedModels += 1
        torch.save(targetNet.state_dict(), f"targetNet{savedModels}.pth")
        with open("highestF1.txt","w") as f:
            f.write(str(highestF1))

    #tockEpoch = time.time()
    #print("Time taken for epoch: ",tockEpoch-tickEpoch)
    plotRewards()

       
print('Complete')
plotRewards(showResult=True)
plt.ioff()
plt.show()