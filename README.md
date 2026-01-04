# Tetris RL - Apprentissage par Renforcement

Agent intelligent qui apprend à jouer à Tetris en utilisant une approximation linéaire de la fonction Q.

## Architecture

```
├── matris_core.py     # Environnement Tetris (états, actions, rewards)
├── tetrominoes.py     # Définition des 7 pièces
├── Qlearning.py       # Entraînement de l'agent (Learn mode)
├── ai_play.py         # Visualisation de l'agent (Play mode)
├── plot_training.py   # Courbe d'apprentissage
├── weights.pkl        # Poids sauvegardés
└── training_stats.pkl # Historique d'entraînement
```

## Installation

```bash
pip install pygame numpy matplotlib
```

## Utilisation

### Learn mode (entraînement)
```bash
python Qlearning.py
```

### Play mode (visualisation)
```bash
python ai_play.py
```

### Afficher la courbe d'apprentissage
```bash
python plot_training.py
```

## Approche

### Problème du Q-learning classique
L'espace d'états de Tetris est immense (~10^61 configurations). Une Q-table classique ne peut pas mémoriser tous ces états.

### Solution : Approximation linéaire de Q

Au lieu de mémoriser Q(s,a) pour chaque état, on l'approxime avec une fonction linéaire :

```
Q(s,a) = w₁·lignes + w₂·trous + w₃·bumpiness + w₄·hauteur
```

Les 4 features :
- **lines** : nombre de lignes complétées
- **holes** : cellules vides sous des blocs (à éviter)
- **bumpiness** : irrégularité de la surface (à éviter)
- **sum_heights** : hauteur totale du plateau (à éviter)

### Sélection d'action

Pour chaque pièce, l'agent :
1. Simule tous les placements possibles (~40)
2. Calcule Q(s,a) pour chacun
3. Choisit le meilleur (avec ε-exploration pendant l'entraînement)

### Optimisation des poids

Les poids sont ajustés par hill-climbing :
- Perturbation aléatoire des poids
- Conservation si amélioration des performances

## Récompenses

```python
reward = lignes × 100 + 1 - game_over × 500
```

- +100 par ligne complétée
- +1 de survie par pièce posée
- -500 si game over

## Contrôles (Play mode)

| Touche | Action |
|--------|--------|
| SPACE | Pause |
| UP/DOWN | Ajuster vitesse |
| ESC | Quitter |

## Performances

Avec les poids par défaut :
- Moyenne : ~8 lignes par partie
- Maximum : ~15 lignes

## Auteur

Projet réalisé dans le cadre du cours "Apprentissage par Renforcement" - M2 2025-2026
