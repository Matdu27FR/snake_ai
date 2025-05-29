import gymnasium as gym
import os
import sys
import numpy as np
from gymnasium import spaces
from src.core.game import game

# Add root folder to import path
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

        # Number of channels: 11 (body, head, apple, behind head, 4 directions, dx, dy, danger)
        self.num_channels = 11

        # History of the last 4 states
        self.history_length = 4
        # Create 4 arrays of zeros with size 1100 (11 channels * 10x10 grid)
        self.history = [np.zeros((self.num_channels * grid_size[0] * grid_size[1],), dtype=np.float32) for _ in range(self.history_length)]

        # Define action space
        self.action_space = spaces.Discrete(4)

        # Define flattened observation space (including history)
        # Total size is 4400 (11 channels * 10x10 grid * 4 history states)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_channels * grid_size[0] * grid_size[1] * self.history_length,),
            dtype=np.float32
        )

        # Calculate maximum possible distance in the grid (diagonal)
        self.max_distance = np.linalg.norm(np.array(grid_size))
        self.previous_food_eaten = len(self.game.snake)
        self.max_steps_without_food = max_steps_without_food
        self.steps_without_food = 0
        self.visited_positions = set()
        self.apples_eaten = 0

    def seed(self, seed=None):
        # Set random seed to reproduce the same sequences
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

        # Reset history with zeros
        self.history = [np.zeros_like(self.history[0]) for _ in range(self.history_length)]

        observation = self._get_observation()
        self._update_history(observation)

        # Return observation and an empty dictionary (Gymnasium format)
        return self._get_combined_observation(), {}

    def step(self, action):
        # Calculate distance between head and food before movement
        old_distance = np.linalg.norm([
            self.game.snakehead[0] - self.game.food[0],
            self.game.snakehead[1] - self.game.food[1]
        ])

        self.game.move(action)

        # Calculate new distance after movement
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
        # Convert game grid to NumPy array
        grid = np.array(self.game.grid)
        head = self.game.snakehead
        food = self.game.food

        # Calculate relative directions to food (normalized)
        dx = (food[0] - head[0]) / self.grid_size[0]
        dy = (food[1] - head[1]) / self.grid_size[1]

        direction = self._get_direction()

        # Create binary channels for game elements
        # Transform cells with value 1 to 1.0, others to 0.0
        body_channel = (grid == 1).astype(np.float32)
        head_channel = (grid == 2).astype(np.float32)
        apple_channel = (grid == 3).astype(np.float32)

        # Channel for segment just behind the head
        behind_head_channel = np.zeros(grid.shape, dtype=np.float32)
        if len(self.game.snake) > 0:
            behind_head = self.game.snake[0]
            behind_head_channel[behind_head[1], behind_head[0]] = 1

        # Channels for current direction (one-hot encoding)
        direction_channels = np.zeros((4, grid.shape[0], grid.shape[1]), dtype=np.float32)
        if 0 <= direction < 4:
            # Fill an entire channel with 1s according to direction
            direction_channels[direction, :, :] = 1

        # Channels for direction to food
        # Fill the entire grid with the same value
        dx_channel = np.full(grid.shape, dx, dtype=np.float32)
        dy_channel = np.full(grid.shape, dy, dtype=np.float32)

        # New vector: danger for each action
        danger_vector = self._get_danger_vector()

        # List of all information channels
        channels = [
            body_channel,
            head_channel,
            apple_channel,
            behind_head_channel,
            *direction_channels,  # Unpack the 4 direction channels
            dx_channel,
            dy_channel,
        ]

        # Add danger_vector as an additional channel
        danger_channel = np.zeros(grid.shape, dtype=np.float32)
        danger_channel[0, :4] = danger_vector  # Place danger vector in first row
        channels.append(danger_channel)

        # Flatten all channels into a single 1D vector of size 1100
        flattened_obs = np.concatenate([ch.flatten() for ch in channels])
        return flattened_obs

    def _get_danger_vector(self):
        """Calculate a vector indicating dangerous actions."""
        head_x, head_y = self.game.snakehead
        # Create a vector of 4 zeros for the 4 directions
        danger = np.zeros(4, dtype=np.float32)  # Vector for 4 actions: [up, down, left, right]

        # Check each direction using check_collision
        if self.game.check_collision(head_x, head_y - 1):  # Up
            danger[0] = 1
        if self.game.check_collision(head_x, head_y + 1):  # Down
            danger[1] = 1
        if self.game.check_collision(head_x - 1, head_y):  # Left
            danger[2] = 1
        if self.game.check_collision(head_x + 1, head_y):  # Right
            danger[3] = 1

        return danger

    def _update_history(self, observation):
        """Update history with new observation."""
        self.history.pop(0)  # Remove oldest state
        self.history.append(observation)  # Add new state

    def _get_combined_observation(self):
        """Combine historical observations into one."""
        # Concatenate the 4 historical observations into a single vector of size 4400
        return np.concatenate(self.history)

    def _get_direction(self):
        if len(self.game.snake) == 0:
            return -1

        head_x, head_y = self.game.snakehead
        body_x, body_y = self.game.snake[0]

        if head_x == body_x and head_y > body_y:
            return 0  # Up
        elif head_x == body_x and head_y < body_y:
            return 1  # Down
        elif head_y == body_y and head_x > body_x:
            return 2  # Left
        elif head_y == body_y and head_x < body_x:
            return 3  # Right
        return -1

    def render(self, mode="human"):
        self.game.print_grid()

    def close(self):
        pass