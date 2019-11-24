import torch
import torch.nn as nn
import torch.optim as optim
import tronEnv
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tronUtil

def logModel(loss, reward, iteration, experiment):
    writer = SummaryWriter('runs/' + experiment)
    writer.add_scalar('State_Train/Loss', loss, iteration)
    writer.add_scalar('State_Test/Reward', reward, iteration)
    writer.close()

class State_Trainer():

    def __init__(self):

        # Hyper Parameters
        # self.gamma = 0.9
        # self.final_epsilon = 0.00001
        # self.initial_epsilon = 0.1
        self.minibatch_size = 512
        self.initial_weights_setting = 0.1
        self.learning_rate = 0.0001
        self.grad_clipping = False

        # Training Process Parameters
        self.number_of_iterations = 0
        self.iterations_between_validation = 2000
        self.env_field_size = 12
        self.validation_episodes = 500
        self.replay_memory_size = 10000

        # Other Shared variables
        self.model = State_Net(self.env_field_size * self.env_field_size,3)
        # self.epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)
        # self.epsilon = self.initial_epsilon
        self.cycle_iteration = 0
        self.global_iteration = 0
        self.replay_memory = tronUtil.readMemory('stateMemory')

    def init_weights(self):
        if type(self.model) == nn.Conv2d or type(self.model) == nn.Linear:
            torch.nn.init.uniform_(self.model.weight, -1 * self.initial_weights_setting, self.initial_weights_setting)
            self.model.bias.data.fill_(self.initial_weights_setting)

    def train(self):

        self.model.train()

        # define Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize mean squared error loss
        criterion = nn.MSELoss()

        # instantiate game
        env = tronEnv.Tron(self.env_field_size)

        # get initial state
        state = env.getState()

        losses = []

        self.cycle_iteration = 0

        # main infinite loop
        while self.cycle_iteration < self.iterations_between_validation:

            action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
            action_index = [torch.randint(self.model.number_of_actions, torch.Size([]), dtype=torch.int)][0]
            action[action_index] = 1
            flatState = state.reshape(-1)
            flatAction = action.reshape(-1)
            model_input = torch.cat((flatState,flatAction)).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # get model output
            # model_output = self.model(model_input)

            # get next state and reward
            state_1, reward, terminal = env.step(action)

            state_1_flat = state_1.reshape(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # save transition to replay memory
            self.replay_memory.append((model_input, state_1_flat))

            # if replay memory is full, remove the oldest transition
            if len(self.replay_memory) > self.replay_memory_size:
                self.replay_memory.pop(0)

            # sample random minibatch
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

            # unpack minibatch
            input_batch = torch.cat(tuple(d[0] for d in minibatch))
            output_batch = torch.cat(tuple(d[1] for d in minibatch))

            # get output for the next state
            output_1_batch = self.model(input_batch)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # calculate loss
            # From blog I read
            loss = criterion(output_1_batch, output_batch)
            losses.append(loss.item())

            # do backward pass
            loss.backward()

            # gradient clipping
            if self.grad_clipping:
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)

            optimizer.step()

            # If last state was terminal then reset environment and get state
            # else set state to previous state
            if terminal:
                env.reset()
                state = env.getState()
            else:
                state = state_1

            self.cycle_iteration += 1
            self.global_iteration += 1

        losses = np.array(losses)
        avgLoss = losses.mean()


        return avgLoss

    def infer(self,action,state):

        self.model.eval()

        flatState = state.reshape(-1)
        flatAction = action.reshape(-1)
        model_input = torch.cat((flatState, flatAction)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        output = self.model(model_input)

        return output

    def saveMemory(self):
        tronUtil.saveMemory('stateMemory',self.replay_memory)

    def uploadMemories(self,state,action,state_1):

        flatState = state.reshape(-1)
        flatAction = action.reshape(-1)
        model_input = torch.cat((flatState, flatAction)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        state_1_flat = state_1.reshape(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # save transition to replay memory
        self.replay_memory.append((model_input, state_1_flat))

        # if replay memory is full, remove the oldest transition
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

class State_Net(nn.Module):

    # This network takes an observation of the environment state and an action and returns a predicted
    # state of the environment

    def __init__(self,field,actions):
        super(State_Net, self).__init__()
        self.number_of_actions = 3

        topFeatures = field + actions

        connect_1_2 = int(topFeatures * 1.33)
        connect_2_3 = int(topFeatures * 1.67)
        connect_3_4 = int(topFeatures * 1.33)


        self.fc1 = nn.Linear(in_features=topFeatures, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.6)

        self.fc4 = nn.Linear(in_features=connect_3_4, out_features=field)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)

        return out