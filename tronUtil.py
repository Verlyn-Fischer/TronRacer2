from torch.utils.tensorboard import SummaryWriter

def logModel(loss, reward, iteration, experiment, modelType):
    writer = SummaryWriter('runs/' + experiment)
    writer.add_scalar(modelType + '_Train/Loss', loss, iteration)
    writer.add_scalar(modelType +'_Test/Reward', reward, iteration)
    writer.close()