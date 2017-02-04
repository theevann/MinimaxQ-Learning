import numpy as np


class Soccer:
    '''
    Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]
    '''

    def __init__(self, h=4, w=5, pA=[3, 2], pB=[1, 1], goalPositions=[1, 2], ballOwner=0, drawProbability=0):
        self.h = h
        self.w = w
        self.goalPositions = np.array(goalPositions)
        self.positions = np.array([pA, pB])
        self.initPositions = np.array([pA, pB])
        self.ballOwner = ballOwner
        self.drawProbability = drawProbability

    def restart(self, pA=None, pB=None, ballOwner=None):
        if pA is not None:
            self.initPositions[0] = pA

        if pB is not None:
            self.initPositions[1] = pB

        if ballOwner is None:
            ballOwner = self.choosePlayer()

        self.positions = self.initPositions.copy()
        self.ballOwner = ballOwner

    def play(self, actionA, actionB):
        if np.random.rand() < self.drawProbability:
            return -2
        first = self.choosePlayer()
        actions = [actionA, actionB]
        m1 = self.move(first, actions[first])
        if (m1 >= 0):
            return m1
        return self.move(1 - first, actions[1 - first])

    def move(self, player, action):
        opponent = 1 - player
        newPosition = self.positions[player] + self.actionToMove(action)

        # If it's opponent position
        if (newPosition == self.positions[opponent]).all():
            self.ballOwner = opponent
        # If it's the goal
        elif self.ballOwner is player and self.isInGoal(*newPosition) >= 0:
            return 1 - self.isInGoal(*newPosition)
        # If it's in board
        elif self.isInBoard(*newPosition):
            self.positions[player] = newPosition
        return -1

    def actionToMove(self, action):
        switcher = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [0, 0],
        }
        return switcher.get(action)

    def isInGoal(self, x, y):
        g1, g2 = self.goalPositions
        if (g1 <= y <= g2):
            if x == -1:
                return 1
            elif x == self.w:
                return 0
        return -1

    def isInBoard(self, x, y):
        return (0 <= x < self.w and 0 <= y < self.h)

    def choosePlayer(self):
        return np.random.randint(0, 2)

    def draw(self, positions=None, ballOwner=None):
        positions = self.positions if positions is None else np.array(positions)
        ballOwner = self.ballOwner if ballOwner is None else ballOwner

        board = ''
        for y in range(self.h)[::-1]:
            for x in range(self.w):
                if ([x, y] == positions[0]).all():
                    board += 'A' if ballOwner is 0 else 'a'
                elif ([x, y] == positions[1]).all():
                    board += 'B' if ballOwner is 1 else 'b'
                else:
                    board += '-'
            board += '\n'

        print(board)


if __name__ == '__main__':
    s = soccer()
    s.draw()
    actions = [[0, 1], [0, 4], [1, 3], [1, 0], [1, 0]]
    actions = [[0, 4], [0, 4], [0, 4], [1, 4], [0, 4]]
    for action in actions:
        print s.play(*action)
        s.draw()
