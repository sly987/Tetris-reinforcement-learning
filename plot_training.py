import pickle
import matplotlib.pyplot as plt

# Charger les stats
try:
    with open("training_stats.pkl", "rb") as f:
        stats = pickle.load(f)
except FileNotFoundError:
    print("Erreur: training_stats.pkl non trouvé.")
    print("Lancez d'abord: python Qlearning.py")
    exit(1)

rewards = stats.get('rewards', stats['scores'])  # fallback si ancien format

# Moyenne glissante
WINDOW = 50
avg_rewards = [
    sum(rewards[max(0, i - WINDOW):i + 1]) / (i + 1 if i < WINDOW else WINDOW)
    for i in range(len(rewards))
]

plt.figure()
plt.plot(avg_rewards)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Progression de l'IA Tetris (Q-learning)")
plt.grid(True, alpha=0.3)
plt.savefig('training_plot.png', dpi=150)
plt.show()