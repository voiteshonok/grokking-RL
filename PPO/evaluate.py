import gym
from gym import make
from agent import Agent
from train import Actor
import numpy as np
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


def run_episode(env, agent):
    frames = []
    observation = env.reset()
    done = False
    while not done:
        # Uncomment the line below to visualize the environment
        env.render()
        frames.append(env.render(mode='rgb_array'))
        observation, reward, done, _ = env.step(agent.act(observation))
    env.close()
    return frames

# Function to create an animation from the frames
def create_animation(frames):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = FuncAnimation(fig, update, frames=frames, interval=50)
    plt.show()


agent = Agent()

env = make("Walker2DBulletEnv-v0")

# rewards = evaluate_policy(env, agent, 50)
# print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
frames = run_episode(env, agent)
# create_animation(frames)