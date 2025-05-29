<div align="center">
  <h1>🐍 Snake AI with Reinforcement Learning 🧠</h1>
</div>

## 📚 Libraries Used

- **Stable-Baselines3**: Implementation of reinforcement learning algorithms (PPO)
- **Gymnasium**: Standardized environment for training AI agents
- **NumPy**: Efficient data manipulation and matrix calculations
- **Pygame**: Game visualization
- **OpenCV & Pillow**: Recording gameplay in video/GIF format

## 🎮 Demonstration

### Learning Evolution

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Agent (150K steps)</b></td>
      <td align="center"><b>Agent (900K steps)</b></td>
      <td align="center"><b>Agent (9.75M steps)</b></td>
        <td align="center"><b>Agent (24M steps)</b></td>
      <td align="center"><b>Agent (45M steps)</b></td>
    </tr>
    <tr>
      <td><img src="recordings/snake_150000_steps.gif" width="200"/></td>
      <td><img src="recordings/snake_900000_steps.gif" width="200"/></td>
      <td><img src="recordings/snake_9750000_steps.gif" width="200"/></td>
            <td><img src="recordings/snake_24000000_steps.gif" width="200"/></td>
      <td><img src="recordings/snake_45000000_steps.gif" width="200"/></td>
    </tr>
  </table>
</div>

### Final Model in Action (66M steps)

<div align="center">
  <img src="recordings/snake_66000000_steps.gif"  width="200" style="border: 3px solid gold;"/>
</div>

> After 66 million training steps, the agent achieves a remarkable performance of 44.29 apples on average per game, with an average duration of 471.85 steps. The timeout rate of 4% indicates that the agent becomes so efficient that it sometimes reaches the time limit without collision.

## 💡 Reward System

| Action | Reward | Description |
|--------|------------|-------------|
| Eating an apple | +50 | Main objective of the game |
| Moving closer to the apple | +0.1 | Encourage the agent to search for food |
| Moving away from the apple | -0.1 | Penalize moving away |
| Revisiting a position | -0.1 | The agent is penalized if the snake's head returns to a cell already visited in the current game. The environment keeps track of all positions visited by the head in a set |
| Wall collision | -10 | Penalty for collision with the grid borders |
| Body collision | -10 | Penalty when the snake bites its tail |
| Timeout (300 steps without eating) | -5 | Avoid passive behaviors by limiting the time between two apples |
| Each action | -0.001 | Small penalty at each step to encourage efficiency |

## 🔍 Information Channels (11 total)

The AI "sees" the game through 11 different information channels:

1. **Snake body**: Position of body segments
2. **Snake head**: Position of the head
3. **Apple**: Position of the food
4. **Segment behind the head**: Position of the first body segment (helps determine current direction)
5. **Current direction - Up**: Represented by a matrix filled with 1 when the snake is moving up
6. **Current direction - Down**: Represented by a matrix filled with 1 when the snake is moving down
7. **Current direction - Left**: Represented by a matrix filled with 1 when the snake is moving left
8. **Current direction - Right**: Represented by a matrix filled with 1 when the snake is moving right
9. **X direction to apple**: Normalized horizontal component (-1 to 1)
10. **Y direction to apple**: Normalized vertical component (-1 to 1)
11. **Danger vector**: Binary vector of 4 dimensions indicating dangerous moves (1 = danger, 0 = safe) for up, down, left, and right directions

Each observation contains these 11 channels, and the environment keeps a history of the last 4 observations to allow the agent to perceive movement and game dynamics.

## 📊 Model Performance

| Model (steps) | Apples (average) | Average duration (steps) | Timeout rate |
|----------------|------------------|------------------------|-----------------|
| 150,000        | 3.88             | 26.02                  | 0%              |
| 900,000        | 10.78            | 75.76                  | 0%              |
| 6,750,000      | 14.91            | 107.61                 | 0%              |
| 9,750,000      | 15.84            | 117.87                 | 0%              |
| 15,000,000     | 24.06            | 206.70                 | 0%              |
| 24,000,000     | 29.78            | 274.15                 | 0%              |
| 36,000,000     | 35.55            | 344.30                 | 0%              |
| 45,000,000     | 40.89            | 414.12                 | 0%              |
| 57,000,000     | 46.50            | 499.41                 | 2%              |
| **66,000,000** | **44.29**        | **471.85**             | **4%**          |


## 🚀 Installation

```bash
git clone https://github.com/your-username/snake-v1.git
cd snake-v1
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```

## 🎯 Main Commands
Here are the main commands to use this project:

Play with a trained model
``` bash
# Play with the latest trained model (without parameters)
python scripts/play_snake.py
# Record as GIF
python scripts/play_snake.py --model checkpoints_by_steps\model_66000000_steps --record

# Record as MP4
python scripts/play_snake.py --model checkpoints_by_steps\model_66000000_steps --record --format mp4
````