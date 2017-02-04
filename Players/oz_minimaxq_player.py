import numpy as np
from scipy.optimize import linprog


class OZMinimaxQPlayer:

    def __init__(self, numStates, numActionsA, numActionsB, decay, expl, gamma):
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 1
        self.V = np.ones(numStates)
        self.Q = np.ones((numStates, numActionsA, numActionsB))
        self.pi = np.ones((numStates, numActionsA)) / numActionsA
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB
        self.learning = True

    def chooseAction(self, state, maxAction):
        if self.learning and np.random.rand() < self.expl:
            action = np.random.randint(maxAction)
        else:
            action = self.weightedActionChoice(state, maxAction)
        return action

    def weightedActionChoice(self, state, maxAction):
        rand = np.random.rand()
        cumSumProb = np.cumsum(self.pi[state, :maxAction])
        cumSumProb /= cumSumProb[-1]
        action = 0
        while rand > cumSumProb[action]:
            action += 1
        return action

    def getReward(self, initialState, finalState, actions, reward, maxActions):
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * self.Q[initialState, actionA, actionB] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        self.V[initialState] = self.updatePolicy(initialState, maxActions)
        self.alpha *= self.decay

    def updatePolicy(self, state, maxActions):
        maxA, maxB = maxActions
        c = np.zeros(maxA + 1)
        c[0] = -1
        A_ub = np.ones((maxB, maxA + 1))
        A_ub[:, 1:] = -self.Q[state, :maxA, :maxB].T
        b_ub = np.zeros(maxB)
        A_eq = np.ones((1, maxA + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * maxA

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        self.pi[state] = np.zeros(self.numActionsA)
        self.pi[state, :maxA] = res.x[1:]
        return res.x[0]

    def policyForState(self, state):
        for i in range(self.numActionsA):
            print("Actions %d : %f" % (i, self.pi[state, i]))


def testUpdatePolicy():
    m = minimaxQPlayer(1, 2, 2, 1e-4, 0.2, 0.9)
    m.Q[0] = [[0, 1], [1, 0.5]]
    m.updatePolicy(0)
    print(m.pi)

if __name__ == '__main__':
    testUpdatePolicy()
