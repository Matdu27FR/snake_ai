import os
import sys
import time
import argparse
import numpy as np
import pygame
import cv2
from stable_baselines3 import PPO

# Add parent directory to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Path added to sys.path: {parent_path}")
sys.path.append(parent_path)

from src.rl.snake_env import SnakeEnv
from src.core.snake import SnakeVisualizer

def find_latest_model():
    base_path = "checkpoints_by_steps"
    
    if not os.path.exists(base_path):
        print(f"The folder {base_path} doesn't exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(base_path) if f.endswith(".zip")]
    
    if len(files) == 0:
        print(f"No models found in {base_path}.")
        sys.exit(1)
    
    files.sort(key=lambda f: os.path.getmtime(os.path.join(base_path, f)))
    latest_model = os.path.join(base_path, files[-1])
    print(f"Model loaded: {latest_model}")
    return latest_model

def play_snake(model_path=None, record=False, format="gif"):
    grid_size = (10, 10)
    cell_size = 20
    
    env = SnakeEnv(grid_size)
    
    if not model_path:
        model_path = find_latest_model()
    else:
        model_path = os.path.normpath(model_path)
        
        if not os.path.exists(model_path):
            if model_path.endswith('.zip'):
                model_path = model_path[:-4]
            
            if not os.path.exists(model_path) and not os.path.exists(model_path + '.zip'):
                print(f"Error: Model file {model_path} doesn't exist.")
                base_dir = "checkpoints_by_steps"
                print(f"\nAvailable models in {base_dir}:")
                if os.path.exists(base_dir):
                    for f in os.listdir(base_dir):
                        if f.endswith('.zip'):
                            print(f"  - {f}")
                sys.exit(1)
                
            if os.path.exists(model_path):
                pass
            elif os.path.exists(model_path + '.zip'):
                model_path += '.zip'
                
        print(f"Model loaded: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    visualizer = SnakeVisualizer(grid_size)

    obs, _ = env.reset()
    done = False
    total_apples = 0
    
    frames = []
    video_writer = None
    
    if record:
        os.makedirs("recordings", exist_ok=True)
        model_name = os.path.basename(model_path)
        steps = model_name.split('_')[1]
        
        if format == "mp4":
            output_path = f"recordings/snake_{steps}_steps.mp4"
            frame_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 10, frame_size)
        else:
            output_path = f"recordings/snake_{steps}_steps.gif"

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_apples = info.get("apples_eaten", total_apples)
        print(f"Head: {env.game.snakehead}, Body: {env.game.snake}")

        num_channels = 11
        last_state = obs[-(num_channels * grid_size[0] * grid_size[1]):]
        last_state = last_state.reshape((num_channels, grid_size[0], grid_size[1]))
        visualizer.render(last_state)
        
        if record:
            screen_array = pygame.surfarray.array3d(visualizer.screen)
            screen_array = np.flip(screen_array, axis=2)
            screen_array = screen_array.transpose(1, 0, 2)
            
            if format == "mp4" and video_writer:
                video_writer.write(screen_array)
            else:
                frames.append(screen_array)

        time.sleep(0.1)

    print(f"Total apples eaten: {total_apples}")
    
    if record:
        if format == "mp4" and video_writer:
            video_writer.release()
            print(f"MP4 video saved to {output_path}")
        elif format == "gif" and frames:
            try:
                from PIL import Image
                pil_frames = []
                for frame in frames:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frames.append(Image.fromarray(frame_rgb))
                
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,
                    loop=0
                )
                print(f"GIF saved to {output_path}")
            except ImportError:
                print("PIL is required to create GIFs. Try 'pip install pillow'")
    
    visualizer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Snake with an AI model")
    parser.add_argument("--model", type=str, help="Path to a specific model")
    parser.add_argument("--record", action="store_true", help="Record the game")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Recording format")
    args = parser.parse_args()
    play_snake(model_path=args.model, record=args.record, format=args.format)