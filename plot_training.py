import pickle
import matplotlib.pyplot as plt

# Charger les récompenses
with open("training_rewards.pkl", "rb") as f:
    rewards = pickle.load(f)

# Moyenne glissante
WINDOW = 50
avg_rewards = [
    sum(rewards[max(0, i-WINDOW):i+1]) /
    (i+1 if i < WINDOW else WINDOW)
    for i in range(len(rewards))
]

plt.figure()
plt.plot(avg_rewards)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Progression de l'IA Tetris (Q-learning)")
plt.show()
