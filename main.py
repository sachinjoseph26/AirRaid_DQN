import gymnasium as gym
import torch
from agent import DQNAgent
from trainer import Trainer

# Hyperparameters
learning_rate = 0.0005  # Increased learning rate
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999  # Slower decay for more exploration
batch_size = 64  # Increased batch size
memory_size = 50000  # Increased replay memory size
target_update_frequency = 5000  # More frequent updates
num_episodes = 500
frame_stack_size = 4  # Number of frames to stack

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Air Raid environment
env = gym.make("ALE/AirRaid-v5")
input_shape = (frame_stack_size, 80, 80)  # Adjusted to (C, H, W) after preprocessing
num_actions = env.action_space.n

# Create the DQN agent
agent = DQNAgent(
    input_shape,
    num_actions,
    lr=learning_rate,
    gamma=gamma,
    epsilon=epsilon_start,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    batch_size=batch_size,
    memory_size=memory_size,
    target_update=target_update_frequency,
    device=device
)

# Create the trainer
trainer = Trainer(env, agent, frame_stack_size=frame_stack_size)

# Train the agent
trainer.train(num_episodes=num_episodes)
