import os
import sys
import argparse
from stable_baselines3 import PPO

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Chemin ajouté à sys.path : {parent_path}")
sys.path.append(parent_path)

from src.rl.snake_env import SnakeEnv

def evaluate_model(model_path, num_episodes=100, max_steps=1000):
    if not os.path.exists(model_path):
        print(f"Le fichier de modèle {model_path} est introuvable.")
        return
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {model_path}: {e}")
        return
    
    env = SnakeEnv(grid_size=(10, 10))
    
    total_pommes = 0
    total_steps = 0
    total_episodes = 0
    timeouts = 0

    print(f"Évaluation de {model_path}...")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        pommes = 0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pommes = info.get("apples_eaten", 0)
            steps += 1
            
            if steps >= max_steps:
                timeouts += 1
                break
            
            done = terminated or truncated

        total_pommes += pommes
        total_steps += steps
        total_episodes += 1

    avg_pommes = total_pommes / num_episodes
    avg_steps = total_steps / num_episodes
    timeout_rate = (timeouts / num_episodes) * 100

    print(f"\nRésumé pour {model_path}:")
    print(f"  Moyenne de pommes mangées: {avg_pommes:.2f}")
    print(f"  Moyenne de steps: {avg_steps:.2f}")
    print(f"  Pourcentage de timeouts: {timeout_rate:.1f}%")
    print(f"  Total des parties jouées: {total_episodes}")
    print(f"  Total des steps effectués: {total_steps}\n")

    return avg_pommes, avg_steps, timeout_rate

if __name__ == "__main__":
    # 1. Création d'un parseur d'arguments avec une description
    parser = argparse.ArgumentParser(description="Évaluer les modèles Snake.")
    
    # 2. Définition des arguments acceptés par le script
    parser.add_argument("--folder",
                        type=str,
                        default="checkpoints_by_steps", 
                        help="Dossier contenant les modèles.")
    parser.add_argument("--num_episodes",
                        type=int,
                        default=100, 
                        help="Nombre d'épisodes.")
    parser.add_argument("--max_steps",
                        type=int,
                        default=1000, 
                        help="Nombre maximum de steps par épisode.")
    
    # 3. Analyse des arguments fournis par l'utilisateur
    args = parser.parse_args()
    # 4. Utilisation des arguments dans le code
    checkpoint_dir = args.folder

    if not os.path.exists(checkpoint_dir):
        print(f"Le dossier {checkpoint_dir} est introuvable.")
        sys.exit(1)

    models = []
    for fichier in os.listdir(checkpoint_dir):
        if fichier.endswith(".zip"):
            models.append(fichier)
    
    if not models:
        print(f"Aucun modèle trouvé dans le dossier {checkpoint_dir}.")
        sys.exit(1)

    models.sort(key=lambda x: int(x.split("_")[1]))

    for model in models:
        model_path = os.path.join(checkpoint_dir, model)
        print(f"\nTesting model: {model}")
        evaluate_model(model_path, num_episodes=args.num_episodes, max_steps=args.max_steps)