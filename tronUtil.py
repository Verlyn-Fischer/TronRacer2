from torch.utils.tensorboard import SummaryWriter
import pickle
import os

def logModel(loss, reward, iteration, experiment, modelType):
    writer = SummaryWriter('runs/' + experiment)
    writer.add_scalar(modelType + '_Train/Loss', loss, iteration)
    writer.add_scalar(modelType +'_Test/Reward', reward, iteration)
    writer.close()

def saveMemory(fileName, memory):
    if len(memory) > 0:
        file = open(fileName, 'wb')
        pickle.dump(memory,file)

def readMemory(fileName):
    if os.path.exists(fileName):
        with open(fileName, 'rb') as fileObj:
            memory = pickle.load(fileObj)
        return memory
    else:
        return []