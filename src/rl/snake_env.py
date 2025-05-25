import gymnasium as gym
import os
import sys
import numpy as np
from gymnasium import spaces
from src.core.game import game

# Ajouter le dossier racine au chemin d'importation
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), max_steps_without_food=300):
        self.grid_size = grid_size

        self.game = game(grid_size[0], grid_size[1])
        self.game.init_grid()
        self.game.init_snake()
        self.game.update_snake()

        # Nombre de canaux : 11 (corps, tête, pomme, derrière tête, 4 directions, dx, dy, danger)
        self.num_channels = 11

        # Historique des 4 derniers états
        self.history_length = 4
        # Crée 4 tableaux de zéros de taille 1100 (11 canaux * grille 10x10)
        self.history = [np.zeros((self.num_channels * grid_size[0] * grid_size[1],), dtype=np.float32) for _ in range(self.history_length)]

        # Définition de l'espace d'action
        self.action_space = spaces.Discrete(4)

        # Définition de l'espace d'observation aplatie (incluant l'historique)
        # La taille totale est 4400 (11 canaux * grille 10x10 * 4 états d'historique)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_channels * grid_size[0] * grid_size[1] * self.history_length,),
            dtype=np.float32
        )

        # Calcule la distance maximale possible dans la grille (diagonale)
        self.max_distance = np.linalg.norm(np.array(grid_size))
        self.previous_food_eaten = len(self.game.snake)
        self.max_steps_without_food = max_steps_without_food
        self.steps_without_food = 0
        self.visited_positions = set()
        self.apples_eaten = 0

    def seed(self, seed=None):
        # Fixe la graine aléatoire pour reproduire les mêmes séquences
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.clear_grid()
        self.game.init_snake()
        self.game.update_snake()
        self.previous_food_eaten = len(self.game.snake)
        self.steps_without_food = 0
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.game.snakehead))
        self.apples_eaten = 0

        # Réinitialiser l'historique avec des zéros
        self.history = [np.zeros_like(self.history[0]) for _ in range(self.history_length)]

        observation = self._get_observation()
        self._update_history(observation)

        # Retourne l'observation et un dictionnaire vide (format Gymnasium)
        return self._get_combined_observation(), {}

    def step(self, action):
        # Calcule la distance entre la tête et la nourriture avant le mouvement
        old_distance = np.linalg.norm([
            self.game.snakehead[0] - self.game.food[0],
            self.game.snakehead[1] - self.game.food[1]
        ])

        self.game.move(action)

        # Calcule la nouvelle distance après le mouvement
        new_distance = np.linalg.norm([
            self.game.snakehead[0] - self.game.food[0],
            self.game.snakehead[1] - self.game.food[1]
        ])

        reward = -0.001
        done = False

        food_eaten = len(self.game.snake) > self.previous_food_eaten
        if food_eaten:
            reward += 50
            self.previous_food_eaten = len(self.game.snake)
            self.steps_without_food = 0
            self.apples_eaten += 1
        else:
            if new_distance < old_distance:
                reward += 0.1
            else:
                reward -= 0.1
            self.steps_without_food += 1

        if tuple(self.game.snakehead) in self.visited_positions:
            reward -= 0.1
        else:
            self.visited_positions.add(tuple(self.game.snakehead))

        if len(self.game.snake) > 0 and self.game.snakehead == self.game.snake[0]:
            reward -= 10
            done = True

        if self.game.game_over:
            reward -= 10
            done = True

        if self.steps_without_food >= self.max_steps_without_food:
            reward -= 5
            done = True

        truncated = False
        info = {"apples_eaten": self.apples_eaten}

        observation = self._get_observation()
        self._update_history(observation)

        return self._get_combined_observation(), reward, done, truncated, info

    def _get_observation(self):
        # Convertit la grille du jeu en tableau NumPy
        grid = np.array(self.game.grid)
        head = self.game.snakehead
        food = self.game.food

        # Calcule les directions relatives vers la nourriture (normalisées)
        dx = (food[0] - head[0]) / self.grid_size[0]
        dy = (food[1] - head[1]) / self.grid_size[1]

        direction = self._get_direction()

        # Crée des canaux binaires pour les éléments du jeu
        # Transforme les cellules de valeur 1 en 1.0, autres en 0.0
        body_channel = (grid == 1).astype(np.float32)
        head_channel = (grid == 2).astype(np.float32)
        apple_channel = (grid == 3).astype(np.float32)

        # Canal pour le segment juste derrière la tête
        behind_head_channel = np.zeros(grid.shape, dtype=np.float32)
        if len(self.game.snake) > 0:
            behind_head = self.game.snake[0]
            behind_head_channel[behind_head[1], behind_head[0]] = 1

        # Canaux pour la direction actuelle (one-hot encoding)
        direction_channels = np.zeros((4, grid.shape[0], grid.shape[1]), dtype=np.float32)
        if 0 <= direction < 4:
            # Remplit un canal complet avec des 1 selon la direction
            direction_channels[direction, :, :] = 1

        # Canaux pour la direction vers la nourriture
        # Remplit toute la grille avec la même valeur
        dx_channel = np.full(grid.shape, dx, dtype=np.float32)
        dy_channel = np.full(grid.shape, dy, dtype=np.float32)

        # Nouveau vecteur : danger pour chaque action
        danger_vector = self._get_danger_vector()

        # Liste de tous les canaux d'information
        channels = [
            body_channel,
            head_channel,
            apple_channel,
            behind_head_channel,
            *direction_channels,  # Déplie les 4 canaux de direction
            dx_channel,
            dy_channel,
        ]

        # Ajouter le danger_vector comme un canal supplémentaire
        danger_channel = np.zeros(grid.shape, dtype=np.float32)
        danger_channel[0, :4] = danger_vector  # Placer le vecteur danger dans la première ligne
        channels.append(danger_channel)

        # Aplatit tous les canaux en un seul vecteur 1D de taille 1100
        flattened_obs = np.concatenate([ch.flatten() for ch in channels])
        return flattened_obs

    def _get_danger_vector(self):
        """Calcule un vecteur indiquant les actions dangereuses."""
        head_x, head_y = self.game.snakehead
        # Crée un vecteur de 4 zéros pour les 4 directions
        danger = np.zeros(4, dtype=np.float32)  # Vecteur pour les 4 actions : [haut, bas, gauche, droite]

        # Vérifier chaque direction en utilisant check_collision
        if self.game.check_collision(head_x, head_y - 1):  # Haut
            danger[0] = 1
        if self.game.check_collision(head_x, head_y + 1):  # Bas
            danger[1] = 1
        if self.game.check_collision(head_x - 1, head_y):  # Gauche
            danger[2] = 1
        if self.game.check_collision(head_x + 1, head_y):  # Droite
            danger[3] = 1

        return danger

    def _update_history(self, observation):
        """Met à jour l'historique avec la nouvelle observation."""
        self.history.pop(0)  # Supprime le plus ancien état
        self.history.append(observation)  # Ajoute le nouvel état

    def _get_combined_observation(self):
        """Combine les observations historiques en une seule."""
        # Concatène les 4 observations historiques en un seul vecteur de taille 4400
        return np.concatenate(self.history)

    def _get_direction(self):
        if len(self.game.snake) == 0:
            return -1

        head_x, head_y = self.game.snakehead
        body_x, body_y = self.game.snake[0]

        if head_x == body_x and head_y > body_y:
            return 0  # Haut
        elif head_x == body_x and head_y < body_y:
            return 1  # Bas
        elif head_y == body_y and head_x > body_x:
            return 2  # Gauche
        elif head_y == body_y and head_x < body_x:
            return 3  # Droite
        return -1

    def render(self, mode="human"):
        self.game.print_grid()

    def close(self):
        pass