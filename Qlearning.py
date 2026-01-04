# Qlearning.py
# Agent Tetris avec évaluation heuristique des actions
# Approche: évaluer chaque placement possible et choisir le meilleur

import numpy as np
import random
import pickle
from matris_core import MatrisCore

# --- Poids des features (à optimiser) ---
# Ces poids déterminent la stratégie de l'agent
WEIGHTS = {
    'lines': 0.76,  # Bonus pour lignes effacées
    'holes': -0.36,  # Pénalité pour trous
    'bumpiness': -0.18,  # Pénalité pour irrégularité
    'sum_heights': -0.51  # Pénalité pour hauteur totale
}

# --- Hyperparamètres d'apprentissage ---
EPISODES = 1000
LEARNING_RATE = 0.01
EPSILON_START = 0.3
EPSILON_END = 0.01

WEIGHTS_FILE = "weights.pkl"
STATS_FILE = "training_stats.pkl"


def evaluate_action(env, action, weights):
    """Évalue une action en simulant son résultat"""
    metrics = env.simulate_action(action)
    score = (
            weights['lines'] * metrics['lines'] +
            weights['holes'] * metrics['holes'] +
            weights['bumpiness'] * metrics['bumpiness'] +
            weights['sum_heights'] * metrics['sum_heights']
    )
    return score, metrics


def choose_action(env, weights, epsilon=0):
    """Choisit la meilleure action (epsilon-greedy)"""
    valid_actions = env.get_valid_actions()

    if not valid_actions:
        return None, None

    if random.random() < epsilon:
        action = random.choice(valid_actions)
        score, metrics = evaluate_action(env, action, weights)
        return action, metrics

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


def train():
    """Entraînement par hill-climbing sur les poids"""
    print("=== Entraînement de l'agent Tetris ===\n")

    # Charger les poids existants ou utiliser les défauts
    try:
        with open(WEIGHTS_FILE, "rb") as f:
            weights = pickle.load(f)
        print(f"Poids chargés: {weights}")
    except FileNotFoundError:
        weights = WEIGHTS.copy()
        print(f"Poids initiaux: {weights}")

    best_weights = weights.copy()
    best_avg_lines = 0

    stats = {'episodes': [], 'lines': [], 'scores': []}

    env = MatrisCore()

    for episode in range(EPISODES):
        # Epsilon decay
        epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / EPISODES)

        # Perturber légèrement les poids pour exploration
        if episode > 0 and episode % 50 == 0:
            for key in weights:
                weights[key] += random.gauss(0, 0.05)

        env.reset()
        pieces = 0

        while not env.done and pieces < 500:
            action, metrics = choose_action(env, weights, epsilon)
            if action is None:
                break
            env.step(action)
            pieces += 1

        stats['episodes'].append(episode)
        stats['lines'].append(env.lines)
        stats['scores'].append(env.score)

        # Affichage
        if (episode + 1) % 50 == 0:
            avg_lines = np.mean(stats['lines'][-50:])
            max_lines = max(stats['lines'][-50:])
            print(f"Episode {episode + 1:4d}/{EPISODES} | "
                  f"Avg lines: {avg_lines:6.1f} | "
                  f"Max: {max_lines:4d} | "
                  f"Epsilon: {epsilon:.3f}")

            # Garder les meilleurs poids
            if avg_lines > best_avg_lines:
                best_avg_lines = avg_lines
                best_weights = weights.copy()

    print(f"\n=== Entraînement terminé ===")
    print(f"Meilleure moyenne: {best_avg_lines:.1f} lignes")
    print(f"Meilleurs poids: {best_weights}")

    # Sauvegarder
    with open(WEIGHTS_FILE, "wb") as f:
        pickle.dump(best_weights, f)
    print(f"Poids sauvegardés dans {WEIGHTS_FILE}")

    with open(STATS_FILE, "wb") as f:
        pickle.dump(stats, f)
    print(f"Stats sauvegardées dans {STATS_FILE}")

    return best_weights


def test(weights, num_games=10):
    """Teste l'agent avec les poids donnés"""
    print(f"\n=== Test sur {num_games} parties ===\n")

    env = MatrisCore()
    results = []

    for game in range(num_games):
        env.reset()
        pieces = 0

        while not env.done and pieces < 1000:
            action, _ = choose_action(env, weights, epsilon=0)
            if action is None:
                break
            env.step(action)
            pieces += 1

        results.append({'lines': env.lines, 'score': env.score, 'pieces': pieces})
        print(f"Game {game + 1}: {env.lines} lignes, score {env.score}, {pieces} pièces")

    avg_lines = np.mean([r['lines'] for r in results])
    max_lines = max([r['lines'] for r in results])
    print(f"\nMoyenne: {avg_lines:.1f} lignes | Max: {max_lines} lignes")

    return results


if __name__ == "__main__":
    # Entraîner
    best_weights = train()

    # Tester
    test(best_weights, num_games=10)