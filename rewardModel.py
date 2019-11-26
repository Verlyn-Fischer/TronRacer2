import torch
import torch.nn as nn
import torch.optim as optim
import tronEnv
import random
import numpy as np
import tronUtil


class Reward_Trainer():

    def __init__(self):

        # Hyper Parameters
        self.gamma = 0.9
        self.final_epsilon = 0.00001
        self.initial_epsilon = 0.1
        self.minibatch_size = 512
        self.initial_weights_setting = 0.1
        self.learning_rate = 0.0001
        self.grad_clipping = False

        # Training Process Parameters
        self.number_of_iterations = 0
        self.iterations_between_validation = 1000
        self.env_field_size = 12
        self.validation_episodes = 500
        self.replay_memory_size = 10000

        # Other Shared variables
        self.model = Reward_Net()
        self.epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)
        self.epsilon = self.initial_epsilon
        self.cycle_iteration = 0
        self.global_iteration = 0
        self.replay_memory = tronUtil.readMemory('rewardMemory')

    def infer(self,state):
        self.model.eval()
        return self.model(state)

    def init_weights(self):
        if type(self.model) == nn.Conv2d or type(self.model) == nn.Linear:
            torch.nn.init.uniform_(self.model.weight, -1 * self.initial_weights_setting, self.initial_weights_setting)
            self.model.bias.data.fill_(self.initial_weights_setting)

    def rewardEstimate(self, state_trainer, state, maxDepth, currentDepth, gamma):

        rewards = self.infer(state)

        if currentDepth < maxDepth:
            action1 = torch.tensor([[1., 0., 0.]])
            action2 = torch.tensor([[0., 1., 0.]])
            action3 = torch.tensor([[0., 0., 1.]])
            state1 = state_trainer.infer(action1, state)
            state2 = state_trainer.infer(action2, state)
            state3 = state_trainer.infer(action3, state)
            state1 = state1.reshape(self.env_field_size, self.env_field_size).unsqueeze(0).unsqueeze(0)
            state2 = state2.reshape(self.env_field_size, self.env_field_size).unsqueeze(0).unsqueeze(0)
            state3 = state3.reshape(self.env_field_size, self.env_field_size).unsqueeze(0).unsqueeze(0)
            bestReward1, bestRewardIndex1 = self.rewardEstimate(state_trainer, state1, maxDepth, currentDepth + 1, gamma)
            bestReward2, bestRewardIndex2 = self.rewardEstimate(state_trainer, state2, maxDepth, currentDepth + 1, gamma)
            bestReward3, bestRewardIndex3 = self.rewardEstimate(state_trainer, state3, maxDepth, currentDepth + 1, gamma)
            reward1 = rewards[0,0].item() + gamma * bestReward1
            reward2 = rewards[0,1].item() + gamma * bestReward2
            reward3 = rewards[0,2].item() + gamma * bestReward3
            bestRewards = torch.tensor([reward1,reward2,reward3])
            bestReward = torch.max(bestRewards).item()
            bestRewardIndex = torch.argmax(bestRewards).item()
            return bestReward, bestRewardIndex

        else:
            bestRewardIndex = torch.argmax(rewards).item()
            bestReward = torch.max(rewards).item()
            return bestReward, bestRewardIndex

    def train_withState(self,state_trainer):

        ##### model.train() and model.eval() ... Need to place where appropriate

        # define Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize mean squared error loss
        criterion = nn.MSELoss()

        # instantiate game
        env = tronEnv.Tron(self.env_field_size)

        # get initial state
        s1 = env.getState()

        losses = []

        self.cycle_iteration = 0

        # main infinite loop
        while self.cycle_iteration < self.iterations_between_validation:

            currentDepth = 0
            maxDepth = 2

            bestReward, bestRewardIndex = self.rewardEstimate(state_trainer,s1, maxDepth, currentDepth, self.gamma)

            #Initialize Action and Set Selection
            action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
            action[bestRewardIndex] = 1

            # get next state and reward
            s2, reward, terminal = env.step(action)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            self.replay_memory.append((s1, action, reward, s2, terminal))

            # if replay memory is full, remove the oldest transition
            if len(self.replay_memory) > self.replay_memory_size:
                self.replay_memory.pop(0)

            # sample random minibatch
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            # get output for the next state
            output_1_batch = self.model(state_1_batch)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))

            # extract Q-value
            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            # From blog I read
            loss = criterion(q_value, y_batch)
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
                s1 = env.getState()
            else:
                s1 = s2

            self.cycle_iteration += 1
            self.global_iteration += 1

        losses = np.array(losses)
        avgLoss = losses.mean()

        return avgLoss

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

            # get output from the neural network
            output = self.model(state)[0]

            # initialize action
            action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # epsilon greedy exploration
            random_action = random.random() <= self.epsilon

            action_index = [torch.randint(self.model.number_of_actions, torch.Size([]),
                                          dtype=torch.int) if random_action else torch.argmax(output)][0]

            # Preparing the input feature set to the state model
            # current_state = state.clone().detach()
            # current_action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
            # current_action[action_index] = 1

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()

            action[action_index] = 1

            # get next state and reward
            state_1, reward, terminal = env.step(action)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            self.replay_memory.append((state, action, reward, state_1, terminal))

            # if replay memory is full, remove the oldest transition
            if len(self.replay_memory) > self.replay_memory_size:
                self.replay_memory.pop(0)

            # epsilon annealing
            self.epsilon = self.epsilon_decrements[self.global_iteration]

            # sample random minibatch
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # get output for the next state
            output_1_batch = self.model(state_1_batch)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))

            # extract Q-value
            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            # From blog I read
            loss = criterion(q_value, y_batch)
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

    def test(self):

        # Set model to testing mode
        self.model.eval()

        env = tronEnv.Tron(self.env_field_size)
        rewards_list = []

        max_iterations = self.env_field_size * self.env_field_size

        max_episodes = self.validation_episodes
        current_episode = 0

        while current_episode < max_episodes:

            episodeReward = 0
            env.reset()
            state = env.getState()

            for iteration in range(max_iterations):

                # get output from the neural network
                output = self.model(state)[0]

                # create blank action
                action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
                if torch.cuda.is_available():  # put on GPU if CUDA is available
                    action = action.cuda()

                # get action from model
                action_index = torch.argmax(output)
                if torch.cuda.is_available():  # put on GPU if CUDA is available
                    action_index = action_index.cuda()
                action[action_index] = 1

                # get next state
                state_1, reward, terminal = env.step(action)

                if not terminal:
                    episodeReward += reward
                    state = state_1
                else:
                    rewards_list.append(episodeReward)
                    break

            current_episode += 1

        rewards_list = np.array(rewards_list)
        avg_rewards = rewards_list.mean()

        return avg_rewards

    def saveMemory(self):
        tronUtil.saveMemory('rewardMemory',self.replay_memory)

class Reward_Net(nn.Module):

    # This network takes an observation of the enviornment state and returns a predicted reward for each of the
    # possible actions

    def __init__(self):
        super(Reward_Net, self).__init__()

        self.number_of_actions = 3

        # Convolution parameters:
        #   input channels
        #   output channels
        #   filter size (square if a single number)
        #   stride (square if a single number)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, stride=1, padding=1, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, stride=1, padding=1, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, stride=1, padding=1, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc4 = nn.Linear(in_features=540, out_features=220)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.6)
        self.fc5 = nn.Linear(in_features=220, out_features=self.number_of_actions)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout1(out)
        out = self.fc5(out)

        return out