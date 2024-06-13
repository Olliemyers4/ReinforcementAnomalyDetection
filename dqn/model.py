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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.nextState)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.nextState
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policyNet(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCHSIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = targetNet(non_final_next_states)[-1].unsqueeze(0).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
    optimiser.step()

def selectAction(state, policyNet, device, stepsDone, EPSSTART, EPSEND, EPSDECAY):
    
    sample = random.random()
    eps_threshold = EPSEND + (EPSSTART - EPSEND) * \
        math.exp(-1. * stepsDone / EPSDECAY)
    stepsDone += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            return policyNet(state)[-1].unsqueeze(0).max(1).indices.view(1, 1),stepsDone
            #return policyNet(state).max(1).indices.view(1, 1)
    else:
        action = random.choice([0, 1])
        return torch.tensor([[action]], device=device, dtype=torch.long),stepsDone


Transition = namedtuple('Transition',
                        ('state', 'action', 'nextState', 'reward','correctAction'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batchSize):

        #Sample with weight to correct action == 1 as it is the minority class
        sampleWeights = []
        for i in range(len(self.memory)):
            if self.memory[i].correctAction == 1:
                sampleWeights.append(0.75)
            else:
                sampleWeights.append(0.1)
        sampleWeights = np.array(sampleWeights)
        sampleWeights = sampleWeights / sampleWeights.sum()
        return random.choices(self.memory, weights=sampleWeights, k=batchSize)


        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, nObservations, nActions):
        super(DQN, self).__init__()
        self.layer1 = nn.LSTM(nObservations, 64,2,batch_first=True)
        self.layer2 = nn.Linear(64, nActions)

    # Called with either one element to determine next action, or a batch
    # during optimisation. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        output,_ = self.layer1(x)
        output = self.layer2(output)
        out = torch.nn.functional.sigmoid(output)
        return out
    

