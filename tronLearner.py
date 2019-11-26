import torch
import torch.nn as nn
import torch.optim as optim
import tronEnv
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import rewardModel
import stateModel
import tronUtil
import os


def trainingOrchestrator(config):

    starting_cycle = config['StartingCycle']
    training_cycles = config['NumberCycles']
    experiment = config['Experiment']
    model_path = config['StartingRewardModel']

    if config['TrainingMode'] == 'StateOnly':
        trainer = config['StateTrainer']

        if os.path.exists(config['StartingStateModel']):
            trainer.trainer.model = torch.load(config['StartingStateModel'], map_location='cpu').eval()
        else:
            trainer.init_weights()

        for cycle in range(starting_cycle, starting_cycle + training_cycles):
            average_loss = trainer.train()
            print(f'State Cycle {cycle} of {training_cycles}')
            tronUtil.logModel(average_loss, 0, cycle, experiment, 'State')
            torch.save(trainer.model, "state_models/current_model_" + str(cycle) + ".pth")

    elif config['TrainingMode']  == 'RewardOnly':

        trainer = config['RewardTrainer']

        if os.path.exists(config['StartingRewardModel']):
            trainer.trainer.model = torch.load(config['StartingRewardModel'], map_location='cpu').eval()
        else:
            trainer.init_weights()

        trainer.global_iteration = trainer.iterations_between_validation * starting_cycle
        trainer.number_of_iterations = (starting_cycle + training_cycles) * trainer.iterations_between_validation
        trainer.epsilon_decrements = np.linspace(trainer.initial_epsilon, trainer.final_epsilon,
                                                 trainer.number_of_iterations)

        for cycle in range(training_cycles):
            average_loss = trainer.train()
            average_reward = trainer.test()
            print(f'Reward Cycle {cycle} of {training_cycles}')
            tronUtil.logModel(average_loss, average_reward, cycle, experiment, 'Reward')
            torch.save(trainer.model, "reward_models/current_model_" + str(cycle) + ".pth")

    elif config['TrainingMode'] == 'Co-train':

        reward_trainer = config['RewardTrainer']
        state_trainer = config['StateTrainer']

        if os.path.exists(config['StartingRewardModel']):
            reward_trainer.trainer.model = torch.load(config['StartingRewardModel'], map_location='cpu').eval()
        else:
            reward_trainer.init_weights()

        if os.path.exists(config['StartingStateModel']):
            state_trainer.trainer.model = torch.load(config['StartingStateModel'], map_location='cpu').eval()
        else:
            state_trainer.init_weights()

        # Set Reward Trainer Control Variables
        reward_trainer.global_iteration = reward_trainer.iterations_between_validation * starting_cycle
        reward_trainer.number_of_iterations = (starting_cycle + training_cycles) * reward_trainer.iterations_between_validation
        reward_trainer.epsilon_decrements = np.linspace(reward_trainer.initial_epsilon, reward_trainer.final_epsilon,
                                                 reward_trainer.number_of_iterations)

        for cycle in range(starting_cycle, starting_cycle + training_cycles):

            print(f'Cycle {cycle} of {starting_cycle + training_cycles}')

            # Train Rewards
            average_loss_reward = reward_trainer.train_withState(state_trainer)
            print('   Reward Training Using State Complete')
            average_reward = reward_trainer.test()
            print('   Reward Test Complete')
            tronUtil.logModel(average_loss_reward, average_reward, cycle, experiment, 'Reward')
            torch.save(reward_trainer.model, "reward_models/current_model_" + str(cycle) + ".pth")

            # Transfer Memory

            someMemories = random.sample(reward_trainer.replay_memory,min(5000,len(reward_trainer.replay_memory)))
            for mem in someMemories:
                state_trainer.uploadMemories(mem[0],mem[1],mem[3])
            print('   Memory Transfer')

            # Train State
            average_loss_state = state_trainer.train()
            print('   State Training Complete')
            tronUtil.logModel(average_loss_state, 0, cycle, experiment, 'State')
            torch.save(state_trainer.model, "state_models/current_model_" + str(cycle) + ".pth")

def main():

    reward_trainer = rewardModel.Reward_Trainer()
    state_trainer = stateModel.State_Trainer()

    trainConfiguration = {'StartingCycle' : 1,
                          'NumberCycles' : 50,
                          'TrainingMode' : 'Co-train',
                          'Exploration' : 'Random',
                          'StartingRewardModel' : 'current_model_X.pth',
                          'StartingStateModel' : 'current_model_X.pth',
                          'Experiment' : 'Co_Train2',
                          'RewardTrainer' : reward_trainer,
                          'StateTrainer': state_trainer}
    # TrainingModes: Co-train, RewardOnly, StateOnly, FixedState
    # Exploration: Random, StateDriven  --- Not used

    trainingOrchestrator(trainConfiguration)

    reward_trainer.saveMemory()
    state_trainer.saveMemory()

main()
