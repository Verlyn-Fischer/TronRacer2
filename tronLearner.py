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


def trainingOrchestrator(config):

    trainer = config['RewardTrainer']
    trainer.init_weights()

    starting_cycle = config['StartingCycle']
    training_cycles = config['NumberCycles']
    experiment = config['Experiment']
    model_path = config['StartingRewardModel']

    if config['TrainingMode'] == 'StateOnly':
        stateTrainer = config['StateTrainer']

        for cycle in range(training_cycles):
            average_loss = stateTrainer.train()
            print(f'State Cycle {cycle} of {training_cycles}')
            tronUtil.logModel(average_loss, 0, cycle, experiment, 'State')
            torch.save(trainer.model, "state_models/current_model_" + str(cycle) + ".pth")

    else:

        if config['StartingCycle'] == 0:

            trainer.global_iteration = trainer.iterations_between_validation * starting_cycle
            trainer.number_of_iterations = (starting_cycle + training_cycles) * trainer.iterations_between_validation
            trainer.epsilon_decrements = np.linspace(trainer.initial_epsilon, trainer.final_epsilon, trainer.number_of_iterations)

            for cycle in range(training_cycles):
                average_loss = trainer.train()
                average_reward = trainer.test()
                print(f'Reward Cycle {cycle} of {training_cycles}')
                tronUtil.logModel(average_loss, average_reward, cycle, experiment, 'Reward')
                torch.save(trainer.model, "reward_models/current_model_" + str(cycle) + ".pth")
        else:
            trainer.model = torch.load(model_path, map_location='cpu').eval()
            trainer.global_iteration = trainer.iterations_between_validation * starting_cycle
            trainer.number_of_iterations = (starting_cycle + training_cycles) * trainer.iterations_between_validation
            trainer.epsilon_decrements = np.linspace(trainer.initial_epsilon, trainer.final_epsilon, trainer.number_of_iterations)

            for cycle in range(starting_cycle, starting_cycle + training_cycles):
                average_loss = trainer.train()
                average_reward = trainer.test()
                print(f'Reward Cycle {cycle} of {starting_cycle + training_cycles}')
                tronUtil.logModel(average_loss, average_reward, cycle, experiment, 'Reward')
                torch.save(trainer.model, "reward_models/current_model_" + str(cycle) + ".pth")


def main():

    reward_trainer = rewardModel.Reward_Trainer()
    state_trainer = stateModel.State_Trainer()

    trainConfiguration = {'StartingCycle' : 0,
                          'NumberCycles' : 70,
                          'TrainingMode' : 'StateOnly',
                          'Exploration' : 'Random',
                          'StartingRewardModel' : 'none',
                          'StartingStateModel' : 'none',
                          'Experiment' : '_experiment_2',
                          'RewardTrainer' : reward_trainer,
                          'StateTrainer': state_trainer}
    # TrainingModes: Co-train, RewardOnly, StateOnly
    # Exploration: Random, StateDriven

    trainingOrchestrator(trainConfiguration)

main()
