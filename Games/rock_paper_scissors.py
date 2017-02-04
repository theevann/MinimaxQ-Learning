import numpy as np


class RockPaperScissors:

    def __init__(self, P=None):
        if P is None:
            P = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        self.P = np.array(P)

    def restart(self):
        pass

    def play(self, actionA, actionB):
        return 1 if self.P[actionA, actionB] < 0 else 0 if self.P[actionA, actionB] > 0 else -2

    def draw(self, P=None):
        P = self.P if P is None else P
        print P

if __name__ == '__main__':
    s = soccer()
    s.draw()
