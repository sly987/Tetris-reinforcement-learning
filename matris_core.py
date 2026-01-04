# matris_core.py
import numpy as np
import random
from tetrominoes import list_of_tetrominoes, rotate


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
        self.total_reward = 0

    def reset(self):
        self.__init__()
        return self.get_observation()

    def get_observation(self):
        return self.matrix.copy()

    def get_valid_actions(self):
        """Retourne les actions valides (rotation, colonne)"""
        valid_actions = []
        seen_shapes = set()

        for rot in range(4):
            shape = rotate(self.current_tetromino.shape, rot)
            shape_key = tuple(tuple(row) for row in shape)

            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)

            shape_width = len(shape[0])
            for col in range(self.width - shape_width + 1):
                if self._can_place(shape, col):
                    valid_actions.append((rot, col))

        return valid_actions

    def _can_place(self, shape, col):
        for start_row in range(-len(shape), 1):
            if self._fits(shape, start_row, col):
                return True
        return False

    def _fits(self, shape, row, col):
        for i, shape_row in enumerate(shape):
            for j, cell in enumerate(shape_row):
                if cell:
                    y, x = row + i, col + j
                    if x < 0 or x >= self.width or y >= self.height:
                        return False
                    if y >= 0 and self.matrix[y, x] > 0:
                        return False
        return True

    def _drop_position(self, shape, col):
        for row in range(-len(shape), self.height):
            if not self._fits(shape, row + 1, col):
                return row
        return self.height - len(shape)

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True

        rotation, col = action
        shape = rotate(self.current_tetromino.shape, rotation)
        final_row = self._drop_position(shape, col)

        # Placer la pièce
        for i, shape_row in enumerate(shape):
            for j, cell in enumerate(shape_row):
                if cell:
                    y, x = final_row + i, col + j
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.matrix[y, x] = 1
                    elif y < 0:
                        self.done = True

        # Effacer lignes
        lines_cleared = self._clear_lines()
        self.lines += lines_cleared
        self.score += [0, 40, 100, 300, 1200][min(lines_cleared, 4)]

        # Nouvelle pièce
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)

        # Game over check
        shape = self.current_tetromino.shape
        spawn_col = (self.width - len(shape[0])) // 2
        if not self._fits(shape, 0, spawn_col):
            self.done = True

        reward = lines_cleared * 100 + 1 - (500 if self.done else 0)
        self.total_reward += reward

        return self.get_observation(), reward, self.done

    def _clear_lines(self):
        cleared = 0
        new_matrix = np.zeros_like(self.matrix)
        new_row = self.height - 1

        for row in range(self.height - 1, -1, -1):
            if np.all(self.matrix[row, :] > 0):
                cleared += 1
            else:
                new_matrix[new_row, :] = self.matrix[row, :]
                new_row -= 1

        self.matrix = new_matrix
        return cleared

    def simulate_action(self, action):
        """Simule une action et retourne les métriques"""
        rotation, col = action
        shape = rotate(self.current_tetromino.shape, rotation)
        final_row = self._drop_position(shape, col)

        temp_matrix = self.matrix.copy()

        # Placer la pièce
        for i, shape_row in enumerate(shape):
            for j, cell in enumerate(shape_row):
                if cell:
                    y, x = final_row + i, col + j
                    if 0 <= y < self.height and 0 <= x < self.width:
                        temp_matrix[y, x] = 1

        # Lignes complètes
        lines = sum(1 for row in range(self.height) if np.all(temp_matrix[row, :] > 0))

        # Supprimer les lignes pour calculer les autres métriques
        if lines > 0:
            new_matrix = np.zeros_like(temp_matrix)
            new_row = self.height - 1
            for row in range(self.height - 1, -1, -1):
                if not np.all(temp_matrix[row, :] > 0):
                    new_matrix[new_row, :] = temp_matrix[row, :]
                    new_row -= 1
            temp_matrix = new_matrix

        # Calculer métriques
        heights = []
        for c in range(self.width):
            col_data = temp_matrix[:, c]
            filled = np.where(col_data > 0)[0]
            heights.append(self.height - filled[0] if len(filled) > 0 else 0)

        holes = 0
        for c in range(self.width):
            found = False
            for r in range(self.height):
                if temp_matrix[r, c] > 0:
                    found = True
                elif found:
                    holes += 1

        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

        return {
            'lines': lines,
            'holes': holes,
            'bumpiness': bumpiness,
            'max_height': max(heights),
            'sum_heights': sum(heights)
        }

    def get_state_key(self):
        """État simplifié pour Q-learning"""
        piece_idx = list_of_tetrominoes.index(self.current_tetromino)
        heights = []
        for c in range(self.width):
            col_data = self.matrix[:, c]
            filled = np.where(col_data > 0)[0]
            heights.append(self.height - filled[0] if len(filled) > 0 else 0)

        # Hauteurs relatives (normalisées)
        min_h = min(heights)
        rel_heights = tuple(min(h - min_h, 6) for h in heights)

        return (rel_heights, piece_idx)