import os
import sys
import argparse
from stable_baselines3 import PPO

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Path added to sys.path: {parent_path}")
sys.path.append(parent_path)

from src.rl.snake_env import SnakeEnv

def evaluate_model(model_path, num_episodes=100, max_steps=1000):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
    
    env = SnakeEnv(grid_size=(10, 10))
    
    total_apples = 0
    total_steps = 0
    total_episodes = 0
    timeouts = 0

    print(f"Evaluating {model_path}...")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        apples = 0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            apples = info.get("apples_eaten", 0)
            steps += 1
            
            if steps >= max_steps:
                timeouts += 1
                break
            
            done = terminated or truncated

        total_apples += apples
        total_steps += steps
        total_episodes += 1

    avg_apples = total_apples / num_episodes
    avg_steps = total_steps / num_episodes
    timeout_rate = (timeouts / num_episodes) * 100

    print(f"\nSummary for {model_path}:")
    print(f"  Average apples eaten: {avg_apples:.2f}")
    print(f"  Average steps: {avg_steps:.2f}")
    print(f"  Timeout percentage: {timeout_rate:.1f}%")
    print(f"  Total games played: {total_episodes}")
    print(f"  Total steps taken: {total_steps}\n")

    return avg_apples, avg_steps, timeout_rate

if __name__ == "__main__":
    # 1. Creating an argument parser with a description
    parser = argparse.ArgumentParser(description="Evaluate Snake models.")
    
    # 2. Defining arguments accepted by the script
    parser.add_argument("--folder",
                        type=str,
                        default="checkpoints_by_steps", 
                        help="Folder containing the models.")
    parser.add_argument("--num_episodes",
                        type=int,
                        default=100, 
                        help="Number of episodes.")
    parser.add_argument("--max_steps",
                        type=int,
                        default=1000, 
                        help="Maximum number of steps per episode.")
    
    # 3. Parsing arguments provided by the user
    args = parser.parse_args()
    # 4. Using arguments in the code
    checkpoint_dir = args.folder

    if not os.path.exists(checkpoint_dir):
        print(f"Folder {checkpoint_dir} not found.")
        sys.exit(1)

    models = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".zip"):
            models.append(file)
    
    if not models:
        print(f"No models found in folder {checkpoint_dir}.")
        sys.exit(1)

    models.sort(key=lambda x: int(x.split("_")[1]))

    for model in models:
        model_path = os.path.join(checkpoint_dir, model)
        print(f"\nTesting model: {model}")
        evaluate_model(model_path, num_episodes=args.num_episodes, max_steps=args.max_steps)