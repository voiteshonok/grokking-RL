from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 2#128
LEARNING_RATE = 5e-4


class DQN:
    def __init__(self, state_dim, action_dim, device):
        self.steps = 0 # Do not change

        self.device = device

        self.model = nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, action_dim)).to(device)

        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.optim = Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.buffer = deque(maxlen=int(1e6))

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)
        # pass

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = random.sample(self.buffer, BATCH_SIZE)
        return list(zip(*batch))
        # pass
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch
        state = torch.tensor(np.array(state, dtype=np.float32)).to(self.device)
        action = torch.tensor(np.array(action, dtype=np.int)).to(self.device)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32)).to(self.device)
        reward = torch.tensor(np.array(reward, dtype=np.float32)).to(self.device)

        state_action_values = self.model(state).gather(1, action.unsqueeze(1).long())
        true_state_action_values = (self.target_model(next_state).max(1)[0].detach().unsqueeze(
            1) * GAMMA) + reward.unsqueeze(1)

        self.optim.zero_grad()
        print(action)
        print(self.model(state))
        print(self.model(state)[action])
        print('some', state_action_values, true_state_action_values)
        print('mine', self.model(state)[0][action], reward + GAMMA * torch.max(self.target_model(next_state).detach()[0]))
        exit(1)
        # print(self.target_model(next_state))
        loss = F.smooth_l1_loss(state_action_values, true_state_action_values)
        loss.backward()
        self.optim.step()
        # pass
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.model.load_state_dict(copy.deepcopy(self.target_model.state_dict()))
        # pass

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.Tensor(np.array(state)).to(self.device)
        return np.argmax(np.array(self.target_model(state).to('cpu')))

        # return 0

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
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


if __name__ == "__main__":
    env = make("LunarLander-v2")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
