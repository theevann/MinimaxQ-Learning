import numpy as np


class OshiZumo:

    def __init__(self, K, N, drawProbability=0):
        self.K = K  # Middle of board
        self.N = N  # Initial Coins
        self.wrestler = K
        self.coins = np.array([N, N])
        self.drawProbability = drawProbability

    def restart(self, K=None, N=None):
        if K is not None:
            self.K = K
        if N is not None:
            self.N = N

        self.wrestler = self.K
        self.coins = np.array([self.N, self.N])

    def play(self, amountA, amountB):
        # amountA, amountB = actionA + 1, actionB + 1
        if np.random.rand() < self.drawProbability:
            return -2
        self.coins -= [amountA, amountB]
        move = 1 if amountA > amountB else -1 if amountB > amountA else 0
        if not self.isInBoard(self.wrestler + move):
            return self.winner()
        self.wrestler += move
        if (self.coins == 0).all():
            return self.winner()
        return -1

    def isInBoard(self, w):
        return 0 <= w <= 2*self.K

    def winner(self):
        return 0 if self.wrestler > self.K else 1 if self.wrestler < self.K else -2

    def draw(self, coins=None, wrestler=None):
        coins = self.coins if coins is None else np.array(coins)
        wrestler = self.wrestler if wrestler is None else wrestler
        board = 'A - %d$    %d%%    %d$ - B' % (coins[0], wrestler * 100. / (2 * self.K), coins[1])
        print(board)


if __name__ == '__main__':
    game = OshiZumo(5, 10)
    game.draw()
    actions = [[1, 2], [1, 1], [2, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    actions = [[1, 2], [2, 1], [3, 1], [2, 4], [1, 2], [1, 0]]
    actions = [[1, 2], [2, 1], [3, 1], [2, 3], [2, 2], [0, 1]]
    for action in actions:
        print game.play(*action)
        game.draw()
    pass
