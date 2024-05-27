import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.optim as optim
import pandas as pd
import model # model.py

# set up matplotlib
isIpython = 'inline' in matplotlib.get_backend()
if isIpython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") #DEBUG - force CPU


def plotRewards(showResult=False):
    plt.figure(1)
    durationsT = torch.tensor(episodeRewards, dtype=torch.float)
    if showResult:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durationsT.numpy())
    # Take 100 episode averages and plot them too
    if len(durationsT) >= 100:
        means = durationsT.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if isIpython:
        if not showResult:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# BATCHSIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPSSTART is the starting value of epsilon
# EPSEND is the final value of epsilon
# EPSDECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimiser

BATCHSIZE = 128
GAMMA = 0.01
EPSSTART = 0.3 #TODO STOP USING EPSILON GREEDY - keeps getting stuck
EPSEND = 0.00
EPSDECAY = 100
TAU = 0.0005
LR = 1e-4


nActions = 2 # 0th -> no anomaly, 1 -> anomaly

TAG = pd.read_csv("normalised.csv",header=0)
TAG,outcome = TAG.iloc[:,1:7],TAG.iloc[:,7] # split into observations and outcomes
names = TAG.iloc[0].index.values


# Make it episodic - split into episodes of n time steps # With N = 1000 each episode is 1000 time steps - 1 total episode
# Each episode is a sequence of observations with a single outcomes - 1 if at least one of the observations is 1, 0 otherwise

# TAG2 needs to be a 2D list where 1st dimension is the episode and 2nd dimension are the time steps within the episode


steps = 10  # 10 per episode

temp = []
for i in range(0,len(TAG)-steps+1): 
    # This is sketchy as it assumes that len(TAG2) is divisible by N - this code needs to be adapted to the generic case at some point
    temp.append(TAG.iloc[i:i+steps].values)
TAGSplit = temp

# Now need to handle the outcomes
temp = []
for i in range(0,len(outcome)-steps+1):
    # This is sketchy as it assumes that len(TAG2) is divisible by N - this code needs to be adapted to the generic case at some point
    holdingOutcome = outcome.iloc[i:i+steps].values
    if any(holdingOutcome) == 1:
        temp.append(1)
    else:
        temp.append(0)
outcomeSplit = temp


#state is the observation of the environment
state = TAG.iloc[0] #reset the environment and get the initial state - will need to import the data
nObservations = len(state)

policyNet = model.DQN(nObservations, nActions).to(device)
targetNet = model.DQN(nObservations, nActions).to(device)
targetNet.load_state_dict(policyNet.state_dict())

optimiser = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)
memory = model.ReplayMemory(400)


stepsDone = 0

episodeRewards = []


def rewarding(action,iteration):
    if action == outcomeSplit[iteration]: #if correct action
      if action == 0: #if no anomaly
         return 3
      else: #if anomaly
         return 20
    else: #if wrong action
      if action == 0: #says no anomaly but there is
         return 0
      else: #says there is anomaly but there isn't
         return 0

numEpisodes = len(TAGSplit)
epoch = 100 #Do every episode 100 times
for eachEpoch in range(epoch):
    #print("Epoch: ",eachEpoch)
    correctSequentially = 1
    incorrectSequentially = 1
    for iEpisode in range(numEpisodes):

        #print("Episode: ",iEpisode,)
        # Initialize the environment and get its state
        #state = TAG2.iloc[0] #reset the environment and get the initial state - will need to import the data
        # Instead get the start of the episode
     

        episode = TAGSplit[iEpisode]
    
        #state = torch.tensor(pd.Series(episode[0],index=names), dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.tensor(episode, dtype=torch.float32, device=device)
        ##print(state.shape)
        action = model.selectAction(state, policyNet, device, stepsDone, EPSSTART, EPSEND, EPSDECAY)
        reward = rewarding(action.item(),iEpisode) # reward of the episode
        # if reward > 0:
        #     reward = reward * correctSequentially
        #     correctSequentially += 1
        #     incorrectSequentially = 1
        # else:
        #     reward = reward * incorrectSequentially
        #     incorrectSequentially += 1
        #     correctSequentially = 1
        correctAction = 0 if ((action.item() == 0 and reward > 0) or (action.item() == 1 and reward < 0)) else 1
        reward = torch.tensor([reward], device=device)
        # Next state is the next episode
        if iEpisode == numEpisodes-1:            
            nextState = None
            memory.push(state, action,nextState, reward,correctAction)
        else: 
            nextState = torch.tensor(TAGSplit[iEpisode+1], dtype=torch.float32, device=device)
            memory.push(state, action,nextState, reward,correctAction)

      
        model.optimiseModel(memory,BATCHSIZE,GAMMA,policyNet,targetNet,optimiser,device)
        targetNetStateDict = targetNet.state_dict()
        policyNetStateDict = policyNet.state_dict()
        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key]*TAU + targetNetStateDict[key]*(1-TAU)
        targetNet.load_state_dict(targetNetStateDict)
        episodeRewards.append(reward)
        plotRewards()

       
print('Complete')
plotRewards(showResult=True)
plt.ioff()
plt.show()