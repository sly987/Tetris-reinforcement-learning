# ai_play.py
import pygame
import pickle
import numpy as np
from matris_core import MatrisCore
from tetrominoes import list_of_tetrominoes

# --- Configuration ---
BLOCK = 30
WIDTH = 10 * BLOCK + 280
HEIGHT = 22 * BLOCK

# Poids par défaut
DEFAULT_WEIGHTS = {
    'lines': 0.76,
    'holes': -0.36,
    'bumpiness': -0.18,
    'sum_heights': -0.51
}

# Couleurs des pièces
PIECE_COLORS = {
    'blue': (105, 105, 255),
    'yellow': (225, 242, 41),
    'pink': (242, 41, 195),
    'green': (22, 181, 64),
    'red': (204, 22, 22),
    'orange': (245, 144, 12),
    'cyan': (10, 255, 226)
}


def evaluate_action(env, action, weights):
    metrics = env.simulate_action(action)
    score = (
            weights['lines'] * metrics['lines'] +
            weights['holes'] * metrics['holes'] +
            weights['bumpiness'] * metrics['bumpiness'] +
            weights['sum_heights'] * metrics['sum_heights']
    )
    return score, metrics


def choose_best_action(env, weights):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None, None

    best_action = None
    best_score = float('-inf')
    best_metrics = None

    for action in valid_actions:
        score, metrics = evaluate_action(env, action, weights)
        if score > best_score:
            best_score = score
            best_action = action
            best_metrics = metrics

    return best_action, best_metrics


def get_current_metrics(env):
    heights = []
    for c in range(env.width):
        col = env.matrix[:, c]
        filled = np.where(col > 0)[0]
        heights.append(env.height - filled[0] if len(filled) > 0 else 0)

    holes = 0
    for c in range(env.width):
        found = False
        for r in range(env.height):
            if env.matrix[r, c] > 0:
                found = True
            elif found:
                holes += 1

    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    return {
        'holes': holes,
        'bumpiness': bumpiness,
        'max_height': max(heights),
        'sum_heights': sum(heights)
    }


def draw_piece(screen, piece, x, y, block_size=20):
    """Dessine une pièce à la position (x, y)"""
    color = PIECE_COLORS.get(piece.color, (150, 150, 150))
    shape = piece.shape

    for row_idx, row in enumerate(shape):
        for col_idx, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    x + col_idx * block_size,
                    y + row_idx * block_size,
                    block_size - 2,
                    block_size - 2
                )
                pygame.draw.rect(screen, color, rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris AI")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    try:
        with open("weights.pkl", "rb") as f:
            weights = pickle.load(f)
    except FileNotFoundError:
        weights = DEFAULT_WEIGHTS

    env = MatrisCore()

    games_played = 0
    total_lines = 0
    max_lines = 0
    last_reward = 0
    speed = 10
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(60, speed + 5)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 5)

        if not paused and not env.done:
            action, _ = choose_best_action(env, weights)
            if action:
                old_reward = env.total_reward
                env.step(action)
                last_reward = env.total_reward - old_reward

        if env.done:
            games_played += 1
            total_lines += env.lines
            max_lines = max(max_lines, env.lines)
            pygame.time.wait(500)
            env.reset()
            last_reward = 0

        metrics = get_current_metrics(env)

        # --- Dessin ---
        screen.fill((20, 20, 30))

        # Grille (blocs posés)
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(x * BLOCK, y * BLOCK, BLOCK - 1, BLOCK - 1)
                if env.matrix[y, x]:
                    pygame.draw.rect(screen, (100, 150, 255), rect)
                else:
                    pygame.draw.rect(screen, (40, 40, 50), rect)

        x_offset = 10 * BLOCK + 15

        # Current piece (celle que l'IA va poser)
        current_label = small_font.render("CURRENT:", True, (100, 255, 100))
        screen.blit(current_label, (x_offset, 15))
        pygame.draw.rect(screen, (60, 80, 60), (x_offset, 40, 100, 80), 2)
        if not env.done:
            draw_piece(screen, env.current_tetromino, x_offset + 20, 50)

        # Next piece
        next_label = small_font.render("NEXT:", True, (255, 255, 100))
        screen.blit(next_label, (x_offset + 110, 15))
        pygame.draw.rect(screen, (60, 60, 70), (x_offset + 110, 40, 100, 80), 2)
        draw_piece(screen, env.next_tetromino, x_offset + 130, 50)

        # Stats (décalé sous les pièces)
        stats_y = 135
        texts = [
            ("=== SCORE ===", (255, 255, 100)),
            (f"Lines: {env.lines}", (200, 200, 200)),
            (f"Score: {env.score}", (200, 200, 200)),
            ("", None),
            ("=== REWARDS ===", (100, 255, 100)),
            (f"Total: {env.total_reward:.0f}", (200, 200, 200)),
            (f"Last:  {last_reward:+.0f}", (100, 255, 100) if last_reward > 0 else (255, 100, 100)),
            ("", None),
            ("=== PENALITES ===", (255, 100, 100)),
            (f"Holes: {metrics['holes']}", (255, 150, 150)),
            (f"Bumpiness: {metrics['bumpiness']}", (255, 150, 150)),
            (f"Max Height: {metrics['max_height']}", (255, 150, 150)),
            (f"Sum Heights: {metrics['sum_heights']}", (255, 150, 150)),
            ("", None),
            ("=== STATS ===", (150, 150, 255)),
            (f"Games: {games_played}", (200, 200, 200)),
            (f"Avg: {total_lines / max(1, games_played):.1f}", (200, 200, 200)),
            (f"Max: {max_lines}", (200, 200, 200)),
            ("", None),
            (f"Speed: {speed}", (150, 150, 150)),
            ("[SPACE] Pause", (120, 120, 120)),
        ]

        for i, (text, color) in enumerate(texts):
            if text:
                surf = small_font.render(text, True, color)
                screen.blit(surf, (x_offset, stats_y + i * 22))

        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_text, (WIDTH // 2 - 80, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()


if __name__ == "__main__":
    main()