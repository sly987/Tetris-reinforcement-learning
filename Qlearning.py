# Qlearning.py
import numpy as np
import random
import pickle
from matris_core import MatrisCore

# --- Hyperparamètres Q-Learning ---
ALPHA = 0.1        # Taux d'apprentissage
GAMMA = 0.9        # Facteur de discount
EPSILON = 0.1      # Exploration vs exploitation
EPISODES = 500     # Nombre d'épisodes d'entraînement
MAX_STEPS = 1000   # Nombre maximum d'actions par épisode

ACTIONS = [0, 1, 2, 3]  # ['left', 'right', 'rotate', 'down']

Q_TABLE_FILE = "qtable.pkl"
REWARDS_FILE = "training_rewards.pkl"

# --- Charger Q-table si existe ---
try:
    with open(Q_TABLE_FILE, "rb") as f:
        Q_table = pickle.load(f)
    print("Q-table chargée depuis", Q_TABLE_FILE)
except FileNotFoundError:
    Q_table = {}
    print("Nouvelle Q-table initialisée")

# --- Liste des récompenses par épisode ---
episode_rewards = []

# --- Entraînement ---
env = MatrisCore()

for episode in range(EPISODES):
    obs = env.reset()
    state = env.state_to_key(obs)
    total_reward = 0

    for step in range(MAX_STEPS):
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))

        # Epsilon-greedy
        if random.random() < EPSILON:
            action = random.choice(ACTIONS)
        else:
            action = int(np.argmax(Q_table[state]))

        # Appliquer l'action
        next_obs, reward, done = env.step(action)
        next_state = env.state_to_key(next_obs)
        total_reward += reward

        if next_state not in Q_table:
            Q_table[next_state] = np.zeros(len(ACTIONS))

        # Q-learning update
        Q_table[state][action] += ALPHA * (
            reward + GAMMA * np.max(Q_table[next_state]) - Q_table[state][action]
        )

        state = next_state
        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}/{EPISODES} - Score: {env.score}, Total reward: {total_reward}")

# --- Sauvegarder Q-table et rewards ---
with open(Q_TABLE_FILE, "wb") as f:
    pickle.dump(Q_table, f)
print("Q-table sauvegardée dans", Q_TABLE_FILE)

with open(REWARDS_FILE, "wb") as f:
    pickle.dump(episode_rewards, f)
print("Rewards sauvegardés dans", REWARDS_FILE)
