import numpy as np


class OZQPlayer:

    def __init__(self, numStates, numActions, decay, expl, gamma):
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 1
        self.V = np.ones(numStates)
        self.Q = np.ones((numStates, numActions))
        self.pi = np.ones((numStates, numActions)) / numActions
        self.numStates = numStates
        self.numActions = numActions
        self.learning = True

    def chooseAction(self, state, maxAction):
        if self.learning and np.random.rand() < self.expl:
            action = np.random.randint(maxAction)
        else:
            action = np.argmax(self.Q[state, :maxAction])
        return action

    def getReward(self, initialState, finalState, actions, reward, maxActions):
        if not self.learning:
            return
        actionA, actionB = actions
        self.Q[initialState, actionA] = (1 - self.alpha) * self.Q[initialState, actionA] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        bestAction = np.argmax(self.Q[initialState, :maxActions[0]])
        self.pi[initialState] = np.zeros(self.numActions)
        self.pi[initialState, bestAction] = 1
        self.V[initialState] = self.Q[initialState, bestAction]
        self.alpha *= self.decay

    def policyForState(self, state):
        for i in range(self.numActions):
            print("Actions %d : %f" % (i, self.pi[state, i]))


if __name__ == '__main__':
    print('THIS IS A Q LEARNING CLASS')
