import torch
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from agent import DQNAgent
from utils import preprocess_state, FrameStack
import numpy as np
from datetime import datetime

class Trainer:
    def __init__(self, env, agent, log_dir='logs', model_dir='models', frame_stack_size=4):
        self.env = env
        self.agent = agent
        self.frame_stack = FrameStack(num_frames=frame_stack_size, height=80, width=80)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.fig, self.ax = plt.subplots()
        self.img_displayed = None
        plt.ion()

    def show_state(self, state, episode, step, reward):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        elif isinstance(state, np.ndarray):
            pass
        else:
            raise TypeError("State must be either a PyTorch tensor or a NumPy array")

        if len(state.shape) == 4:
            state = state[-1]
        elif len(state.shape) == 3:
            if state.shape[2] == 1:
                state = np.squeeze(state, axis=-1)
        elif len(state.shape) == 2:
            state = np.expand_dims(state, axis=-1)

        if len(state.shape) == 3 and state.shape[2] == 1:
            state = np.squeeze(state, axis=-1)

        if self.img_displayed is None:
            self.img_displayed = self.ax.imshow(state)
        else:
            self.img_displayed.set_data(state)

        self.ax.set_title(f"Episode: {episode}, Step: {step}, Total Reward: {reward}")
        plt.pause(0.001)
        self.fig.canvas.draw_idle()

    def train(self, num_episodes):
        total_time = 0
        ave_rewards = []
        episode_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            try:
                state = preprocess_state(state[0], self.agent.device)
                state = self.frame_stack.reset(state)
            except Exception as e:
                print(f"Error preprocessing state at episode {episode}: {e}")
                continue

            done = False
            total_reward = 0
            time_step = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info, _ = self.env.step(action)
                try:
                    next_state = preprocess_state(next_state, self.agent.device)
                    next_state = self.frame_stack.append(next_state)
                except Exception as e:
                    print(f"Error preprocessing next state at episode {episode}: {e}")
                    break

                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update_policy(self.writer)

                state = next_state
                total_reward += reward
                time_step += 1

            total_time += time_step
            episode_rewards.append(total_reward)
            mean_reward = round(np.mean(episode_rewards[-5:]), 3)
            ave_rewards.append(mean_reward)
            self.writer.add_scalar('Average_Cumulative_Reward/Last_5_Episodes', mean_reward, episode)
            self.agent.update_epsilon()
            self.writer.add_scalar('Training/Episode Reward', total_reward, episode)
            self.writer.add_scalar('Training/Epsilon', self.agent.epsilon, episode)
            self.writer.add_scalar('Training/Number of steps', time_step, episode)

            if episode % 50 == 0:
                torch.save(self.agent.policy_net.state_dict(), f'{self.model_dir}/dqn_airraid_{episode}_10K_update.pth')

            print(f"Episode {episode + 1}/{num_episodes}: Total Reward: {total_reward} Time Step: {time_step} Total Time: {total_time}")

        self.writer.close()
        plt.close(self.fig)

    def evaluate(self, num_eval_episodes):
        total_rewards = []

        for episode in range(num_eval_episodes):
            state = self.env.reset()
            self.env.render()
            try:
                state = preprocess_state(state[0], self.agent.device)
            except Exception as e:
                print(f"Error preprocessing state at eval episode {episode}: {e}")
                continue
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info, _ = self.env.step(action)
                try:
                    next_state = preprocess_state(next_state, self.agent.device)
                except Exception as e:
                    print(f"Error preprocessing next state at eval episode {episode}: {e}")
                    break
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

        average_reward = np.mean(total_rewards)
        self.writer.add_scalar('Evaluation/AverageReward', average_reward, 0)
        print(f"Average Reward over {num_eval_episodes} episodes: {average_reward}")

        self.writer.close()
        plt.close(self.fig)
