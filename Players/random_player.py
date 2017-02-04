import numpy as np


class RandomPlayer:

    def __init__(self, numActions):
        self.numActions = numActions

    def chooseAction(self, state, maxAction=None):
        maxAction = self.numActions if maxAction is None else maxAction
        return np.random.randint(maxAction)

    def getReward(self, initialState, finalState, actions, reward, maxActions=None):
        pass
