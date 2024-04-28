import random
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np




def optimiseModel(memory,BATCHSIZE,GAMMA,policyNet,targetNet,optimiser,device):
    if len(memory) < BATCHSIZE:
        return
    transitions = memory.sample(BATCHSIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
   

    stateBatch = torch.cat(batch.state)
    actionBatch = torch.cat(batch.action)
    newReward = list(map(lambda x : torch.tensor([[x]], dtype=torch.float32, device=device), batch.reward))
    rewardBatch = torch.cat(newReward)

    # Compute Q(sT, a) - the model computes Q(sT), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policyNet
    stateActionValues = policyNet(stateBatch)
   
    #stateactionValues is 128x10x1
    #actionBatch is 128x1
    #so to get the actions we need to apply the action function to the stateActionValues on each of the 128 batches
    #to get the 128x1 values
    splitTensor = stateActionValues.chunk(BATCHSIZE,dim = 0)
    aggreates = []
    actions = []
    for tensor in splitTensor:

        outputOfModel = tensor[0]
        aggreate = torch.mean(outputOfModel, dim=1)
        action = torch.argmax(aggreate)
        actions.append(action)
        aggreates.append(aggreate)

    # aggreates is a list of 128 x 2
    # actions is a list of 128 x 1
    # where we need to compute loss
    actions = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    stateActionValues = torch.tensor(aggreate, device=device, dtype=torch.long).unsqueeze(1)
    stateActionValues = stateActionValues.gather(0,actions)

    # Compute V(s{t+1}) for all next states.
    # Expected values of actions for nonFinalNextStates are computed based
    # on the "older" targetNet; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    #nonFinalMask = [s is not None for s in batch.nextState]
    nonFinalNextStates = batch.nextState
    nonFinalNextStates = np.array(nonFinalNextStates)
    nonFinalNextStates = torch.tensor(nonFinalNextStates, dtype=torch.float32, device=device)
    nextStateValues = np.zeros(BATCHSIZE)

    #todo clean this up
    splitTensorT = nonFinalNextStates.chunk(BATCHSIZE,dim = 0)

    aggreatesT = []
    actionsT = []
    for tensor in splitTensorT:
        #Not evn running through targetNet
        outputOfModel = tensor[0]
        aggreate = torch.mean(outputOfModel, dim=1)
        action = torch.argmax(aggreate)
        actionsT.append(action)
        aggreatesT.append(aggreate)

    # aggreates is a list of 128 x 2
    # actions is a list of 128 x 1
    # where we need to compute loss




    #qValues = agge






    nextActionValues = torch.tensor(actionsT, device=device, dtype=torch.long).unsqueeze(1)
    nextActionValues = nextActionValues.gather(0, actionBatch)
    # Compute the expected Q values
    #    targetQValues = torch.mean(rewardBatch, dim=1) + GAMMA * Mex















    expectedStateActionValues = aggreatesT*GAMMA + rewardBatch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(stateActionValues, expectedStateActionValues)

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
    optimiser.step()


def selectAction(state, policyNet, device, stepsDone, EPSSTART, EPSEND, EPSDECAY):
    sample = random.random()
    epsThreshold = EPSEND + (EPSSTART - EPSEND) * \
        math.exp(-1. * stepsDone / EPSDECAY)
    stepsDone += 1 #remember to update this
    if sample > epsThreshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policyOut = policyNet(state)
            #print(policyOut)
            aggreate = torch.mean(policyOut, dim=1)
            action = np.argmax(aggreate)
           

            # We now have a list of actions for each timestep in the episode
            # we need to return the action that was taken at the current timestep
            # We need to choose one value which should be 1 if any of the values in the list is 1 else 0

    else:
        action = random.choice([0, 1])

    return torch.tensor([[action]], device=device, dtype=torch.long),stepsDone # 0 -> no anomaly, 1 -> anomaly


Transition = namedtuple('Transition',
                        ('state', 'action', 'nextState', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, nObservations, nActions):
        super(DQN, self).__init__()
        #self.layer1 = nn.Linear(nObservations, 64)
        #self.layer2 = nn.Linear(64, 64)
        #self.layer3 = nn.Linear(64, nActions)
        self.layer1 = nn.LSTM(nObservations, 64,2,batch_first=True)
        self.layer2 = nn.Linear(64, nActions)

    # Called with either one element to determine next action, or a batch
    # during optimisation. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        #return self.layer3(x)
        output,_ = self.layer1(x)
        output = self.layer2(output)
        out = torch.nn.functional.sigmoid(output)
        return out
    

