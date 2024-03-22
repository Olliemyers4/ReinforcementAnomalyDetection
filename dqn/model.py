import random
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
import torch
import math



Transition = namedtuple('Transition',
                        ('state', 'action', 'nextState', 'reward'))


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
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.nextState)), device=device, dtype=torch.bool)
    nonFinalNextStates = torch.cat([s for s in batch.nextState
                                                if s is not None])
    stateBatch = torch.cat(batch.state)
    actionBatch = torch.cat(batch.action)
    rewardBatch = torch.cat(batch.reward)

    # Compute Q(sT, a) - the model computes Q(sT), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policyNet
    stateActionValues = policyNet(stateBatch).gather(1, actionBatch)

    # Compute V(s{t+1}) for all next states.
    # Expected values of actions for nonFinalNextStates are computed based
    # on the "older" targetNet; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    nextStateValues = torch.zeros(BATCHSIZE, device=device)
    with torch.no_grad():
        nextStateValues[nonFinalMask] = targetNet(nonFinalNextStates).max(1).values
    # Compute the expected Q values
    expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

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
    stepsDdone += 1 #remember to update this
    if sample > epsThreshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policyNet(state).max(1).indices.view(1, 1),stepsDone
    else:
        return torch.tensor([[random.choice([0, 1])]], device=device, dtype=torch.long),stepsDone # 0 -> no anomaly, 1 -> anomaly


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
        self.layer1 = nn.Linear(nObservations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, nActions)

    # Called with either one element to determine next action, or a batch
    # during optimisation. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

