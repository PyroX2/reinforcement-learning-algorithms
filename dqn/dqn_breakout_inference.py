import torch
from torch import nn
import gymnasium as gym
import math
import numpy as np
import ale_py
import cv2

EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 3000

BATCH_SIZE = 128
MAX_MEMORY_LENGTH = 10_000
GAMMA = 0.99
TAU = 0.005

NUM_STEPS = 1_000_000
LEARNING_RATE = 1e-4
NUM_STEPS_TO_AVERAGE = 10

class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Maps quality of each action at given state
        self.fc = nn.Sequential(nn.Linear(obs_space, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, action_space))

    def forward(self, state):
        return self.fc(state)

class ReplayBuffer:
    def __init__(self, max_size):
        self._max_size = max_size

        # Memory for storing transitions
        self._memory = {"state": [],
                       "action": [],
                       "new_state": [],
                       "reward": []}

    def get_batch(self, batch_size):
        if batch_size > len(self._memory["state"]):
            raise ValueError("Batch size must be smaller than the memory length")

        batch_indices = torch.randperm(len(self._memory["state"]))[:batch_size]

        states = torch.tensor(np.array(self._memory["state"]), dtype=torch.float32)[batch_indices]
        actions = torch.tensor(np.array(self._memory["action"]), dtype=torch.int64)[batch_indices]
        new_states = torch.tensor(np.array(self._memory["new_state"]), dtype=torch.float32)[batch_indices]
        rewards = torch.tensor(np.array(self._memory["reward"]), dtype=torch.float32)[batch_indices]

        return states, actions, new_states, rewards
    
    def update(self, state, action, new_state, reward):
        self._memory["state"].append(state)
        self._memory["action"].append(action)
        self._memory["new_state"].append(new_state)
        self._memory["reward"].append(reward)

        if len(self._memory["state"]) > self._max_size:
            self._memory["state"].pop(0)
            self._memory["action"].pop(0)
            self._memory["new_state"].pop(0)
            self._memory["reward"].pop(0)

    def __len__(self):
        return len(self._memory["state"])

def get_action(state, policy_network, action_space, eps_threshold):
    if torch.rand(1).item() > eps_threshold:
        actions = policy_network(state)
        return torch.argmax(actions).item()
    else:
        return torch.randint(action_space, (1,)).item()

def optimize_model(replay_buffer: ReplayBuffer, 
                   policy_network: torch.nn.Module, 
                   target_network: torch.nn.Module, 
                   optimizer: torch.optim.Adam, 
                   criterion: nn.SmoothL1Loss) -> None:
    optimizer.zero_grad()
    
    state_tensor, action_tensor, new_state_tensor, reward_tensor = replay_buffer.get_batch(BATCH_SIZE)

    # Get current q values for each action taken given a state
    q_values_for_actions_taken = policy_network(state_tensor).gather(1, action_tensor.reshape(-1, 1))

    expected_q_values = GAMMA*torch.max(target_network(new_state_tensor), dim=-1).values+reward_tensor
    # expected_q_values *= reward_tensor

    loss = criterion(q_values_for_actions_taken.squeeze(), expected_q_values)

    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()

env = gym.make("Breakout-v4", render_mode="rgb_array", obs_type='ram')

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n.item()

print(f"Observation space is: {obs_space}")
print(f"Action space is: {action_space}")

policy_network = DQN(obs_space, action_space)
policy_network.load_state_dict(torch.load("dqn_breakout_trained.pth"))

target_network = DQN(obs_space, action_space)
target_network.load_state_dict(policy_network.state_dict())

replay_buffer = ReplayBuffer(MAX_MEMORY_LENGTH)

optimizer = torch.optim.AdamW(policy_network.parameters(), LEARNING_RATE, amsgrad=True)
criterion = nn.SmoothL1Loss()

episode_reward = []

state, info = env.reset()
episode_length = torch.tensor(0, dtype=torch.float64)
for step in range(NUM_STEPS):
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * step / EPS_DECAY)
    eps_threshold = 0  # For inference, we use a greedy policy

    # frame = env.render()
    
    
    action = get_action(torch.tensor(state, dtype=torch.float32), policy_network, action_space, eps_threshold)
    new_state, reward, terminated, truncated, _ = env.step(action)

    episode_length += reward

    if step % 1000 == 0:
        print(f"Step: {step}, action: {action}, reward: {reward}, eps_threshold: {eps_threshold}")

    if terminated or truncated:
        env.reset()
        episode_reward.append(episode_length)
        episode_length = torch.tensor(0, dtype=torch.float64)
        reward = 0
        print(f"Episode finished after {step} steps")

    state = new_state

    if len(episode_reward) > NUM_STEPS_TO_AVERAGE:
        mean_episode_reward = torch.mean(torch.tensor(episode_reward))
        episode_reward = []
        print(f"Step: {step}, mean episode reward: {mean_episode_reward}")

    # cv2.imshow("Breakout", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

torch.save(policy_network.state_dict(), "dqn_breakout_trained.pth")


