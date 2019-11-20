import numpy as np
import random
import torch

class Tron():

    # FIELD SETTINGS
    # Player 1:
    #   Heading Up: 10
    #   Heading Right: 11
    #   Heading Down: 12
    #   Heading Left: 13
    #
    # Player 2:
    #   Heading Up: 20
    #   Heading Right: 21
    #   Heading Down: 22
    #   Heading Left: 23

    # Light Path: 3
    # Empty field: 0

    # ACTION CHOICES
    # Left: 1
    # Straight: 2
    # Right: 3

    # PLAYER DIRECTION
    # Up: 0
    # Right: 1
    # Down: 2
    # Left: 3

    def __init__(self,field):
        self.field = field
        self.actionSpace = np.zeros((field,field))
        self.x = random.randint(3,field-3)
        self.y = random.randint(3,field-3)
        self.direction = random.randint(0,3)
        self.actionSpace[self.x,self.y] = self.direction + 10
        self.done = False

    def reset(self):
        self.actionSpace = np.zeros((self.field,self.field))
        self.x = random.randint(3,self.field-3)
        self.y = random.randint(3,self.field-3)
        self.direction = random.randint(0,3)
        self.actionSpace[self.x,self.y] = self.direction + 10
        self.done = False

    def step(self, action_T):

        if action_T[0] == 1:
            action = 1
        elif action_T[1] == 1:
            action = 2
        elif action_T[2] == 1:
            action = 3

        # Change player direction

        if action == 1:
            self.direction = (self.direction + 1) % 4

        if action == 3:
            self.direction = (self.direction - 1) % 4

        # Get step ahead

        if self.direction == 0:
            xTemp = self.x
            yTemp = self.y + 1

        if self.direction == 1:
            xTemp = self.x - 1
            yTemp = self.y

        if self.direction == 2:
            xTemp = self.x
            yTemp = self.y - 1

        if self.direction == 3:
            xTemp = self.x + 1
            yTemp = self.y

        # Check within bounds and not colliding

        if xTemp <= self.field - 1 and xTemp >= 0 and yTemp <= self.field - 1 and yTemp >= 0:
            if self.actionSpace[xTemp,yTemp] == 0:
                self.actionSpace[self.x,self.y] = 3
                self.x = xTemp
                self.y = yTemp
                self.actionSpace[self.x,self.y] = self.direction + 10
                reward = 1
                done = False
            else:
                reward = -1
                done = True
        else:
            reward = -1
            done = True

        out = self.actionSpace
        out = torch.from_numpy(out)
        out = out.reshape(1, 1, self.field, -1)
        out = out.type(torch.float32)
        observation = out

        return observation, reward, done

    def getState(self):
        out = self.actionSpace
        out = torch.from_numpy(out)
        out = out.reshape(1,1,self.field,-1)
        out = out.type(torch.float32)

        return out

    def action_space_n(self):
        return 3

# def main():
#     tron = Tron(11)
#     for count in range(30):
#         dir =  random.randint(0,3)
#         obs, reward, done, info = tron.step(dir)
#         if done:
#             print('reset')
#             tron.reset()
#
#
# main()
