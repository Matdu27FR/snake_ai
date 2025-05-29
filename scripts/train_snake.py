import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add parent path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Path added to sys.path: {parent_path}")
sys.path.append(parent_path)

from src.rl.snake_env import SnakeEnv
from scripts.save_callback import SaveAndLogCallback

# Limit PyTorch to 12 threads to avoid CPU overload
torch.set_num_threads(12)

# Number of parallel environments
N_ENVS = 12

# Maximum folder size in GB before stopping training
MAX_FOLDER_SIZE_GB = 30

def make_env(rank):
    """Creates a Snake environment with a specific seed."""
    def _init():
        env = SnakeEnv(grid_size=(10, 10))
        env.seed(rank)
        return env
    return _init

def get_folder_size(folder_path):
    """Calculates the total size of a folder in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def check_folder_size_and_warn(folder_path, max_size_gb):
    """Checks if the folder size exceeds the limit."""
    folder_size_bytes = get_folder_size(folder_path)
    folder_size_gb = folder_size_bytes / (1024 ** 3)  # Conversion to GB
    if folder_size_gb > max_size_gb:
        warning_file = os.path.join(folder_path, "warning_30go.txt")
        with open(warning_file, "w") as f:
            f.write(f"Warning: Folder size has exceeded {max_size_gb} GB ({folder_size_gb:.2f} GB).\n")
        print(f"Training stopped: Folder size has exceeded {max_size_gb} GB.")
        exit(1)  # Stop the script

if __name__ == "__main__":
    # Check folder size before starting
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    check_folder_size_and_warn(project_folder, MAX_FOLDER_SIZE_GB)

    # Create logs directory
    logs_dir = os.path.join(project_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Log file for steps
    log_file_steps = os.path.join(logs_dir, "logs_steps.txt")

    # Create multiple parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Folder to save models
    save_path_steps = "checkpoints_by_steps"
    os.makedirs(save_path_steps, exist_ok=True)

    # Save frequencies by steps
    save_freqs_steps = (
        [10_000, 50_000] +
        [i for i in range(100_000, 1_000_001, 50_000)] +
        [i for i in range(1_000_000, 10_000_001, 250_000)] + 
        [i for i in range(10_000_000, 100_000_001, 1_000_000)] + 
        [i for i in range(100_000_000, 1_000_000_001, 10_000_000)]
    )

    # Model initialization
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,  # Collected trajectories
        batch_size=256,  # Mini-batch size
        learning_rate=1e-4,  # Learning rate
        ent_coef=0.01,  # Exploration
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu"  # GPU if available
    )

    # Callback to save by steps
    callback_steps = SaveAndLogCallback(
        save_freqs=save_freqs_steps,
        save_path=save_path_steps,
        log_file=log_file_steps
    )

    model.learn(total_timesteps=1_000_000_000, callback=callback_steps)
    model.save("ppo_snake_final")