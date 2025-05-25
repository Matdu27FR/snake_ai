import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Ajouter le chemin parent
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Chemin ajouté à sys.path : {parent_path}")
sys.path.append(parent_path)

from src.rl.snake_env import SnakeEnv
from scripts.save_callback import SaveAndLogCallback

# Limiter PyTorch à 12 threads pour éviter la surcharge CPU
torch.set_num_threads(12)

# Nombre d'environnements parallèles
N_ENVS = 12

# Taille maximale du dossier en Go avant d'arrêter l'entraînement
MAX_FOLDER_SIZE_GB = 30

def make_env(rank):
    """Crée un environnement Snake avec une graine spécifique."""
    def _init():
        env = SnakeEnv(grid_size=(10, 10))
        env.seed(rank)
        return env
    return _init

def get_folder_size(folder_path):
    """Calcule la taille totale d'un dossier en octets."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def check_folder_size_and_warn(folder_path, max_size_gb):
    """Vérifie si la taille du dossier dépasse la limite."""
    folder_size_bytes = get_folder_size(folder_path)
    folder_size_gb = folder_size_bytes / (1024 ** 3)  # Conversion en Go
    if folder_size_gb > max_size_gb:
        warning_file = os.path.join(folder_path, "warning_30go.txt")
        with open(warning_file, "w") as f:
            f.write(f"Attention : La taille du dossier a dépassé {max_size_gb} Go ({folder_size_gb:.2f} Go).\n")
        print(f"Entraînement arrêté : La taille du dossier a dépassé {max_size_gb} Go.")
        exit(1)  # Arrêter le script

if __name__ == "__main__":
    # Vérifier la taille du dossier avant de commencer
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    check_folder_size_and_warn(project_folder, MAX_FOLDER_SIZE_GB)

    # Créer le répertoire logs
    logs_dir = os.path.join(project_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Fichier de logs pour les steps
    log_file_steps = os.path.join(logs_dir, "logs_steps.txt")

    # Créer plusieurs environnements parallèles
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Dossier pour sauvegarder les modèles
    save_path_steps = "checkpoints_by_steps"
    os.makedirs(save_path_steps, exist_ok=True)

    # Fréquences de sauvegarde par steps
    save_freqs_steps = (
        [10_000, 50_000] +
        [i for i in range(100_000, 1_000_001, 50_000)] +
        [i for i in range(1_000_000, 10_000_001, 250_000)] + 
        [i for i in range(10_000_000, 100_000_001, 1_000_000)] + 
        [i for i in range(100_000_000, 1_000_000_001, 10_000_000)]
    )

    # Initialisation du modèle
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,  # Trajectoires collectées
        batch_size=256,  # Taille des mini-lots
        learning_rate=1e-4,  # Taux d'apprentissage
        ent_coef=0.01,  # Exploration
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu"  # GPU si disponible
    )

    # Callback pour sauvegarder par steps
    callback_steps = SaveAndLogCallback(
        save_freqs=save_freqs_steps,
        save_path=save_path_steps,
        log_file=log_file_steps
    )

    model.learn(total_timesteps=1_000_000_000, callback=callback_steps)
    model.save("ppo_snake_final")