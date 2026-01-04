# matris_core.py
import numpy as np
import random
from tetrominoes import list_of_tetrominoes, rotate

ACTIONS = ['left', 'right', 'rotate', 'down']

class MatrisCore:
    def __init__(self):
        self.width = 10
        self.height = 22
        self.matrix = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.lines = 0
        self.done = False
        self.current_tetromino = random.choice(list_of_tetrominoes)
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.position = (0, 3)
        self.rotation = 0
        self.total_reward = 0

    def reset(self):
        self.__init__()
        return self.get_observation()

    def get_observation(self):
        """Retourne la matrice + horizon (ligne la plus haute occupée)"""
        obs = self.matrix.copy()
        horizon = np.argmax(np.flipud((self.matrix>0).astype(int)), axis=0)
        horizon = self.height - horizon
        return (obs, tuple(horizon))

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True

        if action == 0:  # left
            self.move(-1)
        elif action == 1:  # right
            self.move(1)
        elif action == 2:  # rotate
            self.rotate()
        elif action == 3:  # down
            self.drop()

        self.gravity()

        # reward basé sur la nouvelle configuration
        reward = self.compute_reward()
        self.total_reward += reward
        return self.get_observation(), reward, self.done

    def move(self, dx):
        y, x = self.position
        new_x = x + dx
        if self.fits(self.current_tetromino.shape, (y, new_x)):
            self.position = (y, new_x)

    def rotate(self):
        new_rot = (self.rotation + 1) % 4
        new_shape = rotate(self.current_tetromino.shape, new_rot)
        if self.fits(new_shape, self.position):
            self.rotation = new_rot

    def drop(self):
        y, x = self.position
        while self.fits(self.current_tetromino.shape, (y+1, x)):
            y += 1
        self.position = (y, x)
        self.lock()

    def gravity(self):
        y, x = self.position
        if self.fits(self.current_tetromino.shape, (y+1, x)):
            self.position = (y+1, x)
        else:
            self.lock()

    def fits(self, shape, pos):
        y0, x0 = pos
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    y, x = y0+i, x0+j
                    if x<0 or x>=self.width or y>=self.height:
                        return False
                    if y>=0 and self.matrix[y,x]>0:
                        return False
        return True

    def lock(self):
        y0, x0 = self.position
        shape = rotate(self.current_tetromino.shape, self.rotation)
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    y, x = y0 + i, x0 + j
                    if y >= self.height:  # <-- éviter dépassement
                        self.done = True
                    elif y >= 0:
                        self.matrix[y, x] = 1
                    else:
                        self.done = True
        cleared = self.clear_lines()
        self.score += cleared ** 2
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.position = (0, 3)
        self.rotation = 0

        # 🔥 TEST GAME OVER AU SPAWN
        if not self.fits(self.current_tetromino.shape, self.position):
            self.done = True

    def clear_lines(self):
        cleared = 0
        new_matrix = np.zeros_like(self.matrix)
        new_row = self.height - 1
        for row in range(self.height-1, -1, -1):
            if all(self.matrix[row,:]>0):
                cleared += 1
            else:
                new_matrix[new_row,:] = self.matrix[row,:]
                new_row -= 1
        self.matrix = new_matrix
        return cleared

    def compute_reward(self):
        """Calcule le reward pour l'état actuel"""
        reward = 0

        # 1. Lignes effacées
        cleared = np.sum(np.all(self.matrix > 0, axis=1))
        reward += 10 * cleared  # bonus par ligne



        # 3. Pénalité pour trous (cellules vides sous des blocs)
        # holes = 0
        # for col in range(self.width):
        #     column = self.matrix[:, col]
        #     filled_found = False
        #     for cell in column:
        #         if cell > 0:
        #             filled_found = True
        #         elif filled_found and cell == 0:
        #             holes += 1
        # reward -= 5 * holes

        # 4. Pénalité pour la différence de hauteur (horizon irrégulier)
        heights = [self.height - np.argmax(np.flipud(self.matrix[:, c])) for c in range(self.width)]
        max_diff = max(heights) - min(heights)
        reward -= max_diff

        # 5. Game over
        if self.done:
            reward -= 100

        return reward


    def state_to_key(self, obs):
        _, horizon = obs
        return tuple(horizon)

