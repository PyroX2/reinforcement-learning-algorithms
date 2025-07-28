import torch
from torch import nn
import gymnasium as gym
import math
import numpy as np
import ale_py
import cv2

EPS_START = 0.1
EPS_END = 0.001
EPS_DECAY = 1e6

BATCH_SIZE = 512
MAX_MEMORY_LENGTH = 512
GAMMA = 0.99
TAU = 0.005

NUM_STEPS = 500_000
LEARNING_RATE = 1e-3
NUM_EPISODES_TO_AVERAGE = 4

UPDATE_TARGET_NETWORK_STEPS = 5_000 # number of steps after which target network is updated

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Maps quality of each action at given state
        self.fc = nn.Sequential(nn.Linear(obs_space, 64),
                                nn.ReLU(),
                                nn.Linear(64, action_space))

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

        states = torch.tensor(np.array(self._memory["state"]), dtype=torch.float32).to(device)[batch_indices]
        actions = torch.tensor(np.array(self._memory["action"]), dtype=torch.int64).to(device)[batch_indices]
        new_states = torch.tensor(np.array(self._memory["new_state"]), dtype=torch.float32).to(device)[batch_indices]
        rewards = torch.tensor(np.array(self._memory["reward"]), dtype=torch.float32).to(device)[batch_indices]

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
                   criterion: nn.SmoothL1Loss) -> float:
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

    return loss.item()

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", obs_type='ram')

obs_space = env.observation_space.shape[0]
action_space = env.action_space.n.item()

print(f"Observation space is: {obs_space}")
print(f"Action space is: {action_space}")

policy_network = DQN(obs_space, action_space)
# policy_network.load_state_dict(torch.load("dqn_breakout_trained.pth", weights_only=True))

target_network = DQN(obs_space, action_space)
target_network.load_state_dict(policy_network.state_dict())

policy_network.to(device)
target_network.to(device)

replay_buffer = ReplayBuffer(MAX_MEMORY_LENGTH)

optimizer = torch.optim.AdamW(policy_network.parameters(), LEARNING_RATE, amsgrad=True)
criterion = nn.SmoothL1Loss()

episode_reward_list = []

state, info = env.reset(seed=0)
lives = info['lives'] + 1

loss = 0

episode_reward = torch.tensor(0, dtype=torch.float64)
for step in range(NUM_STEPS):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * step / EPS_DECAY)
    
    if info['lives'] < lives:
        action = torch.tensor(1) # Fire
    else:
        action = get_action(torch.tensor(state, dtype=torch.float32).to(device), policy_network, action_space, eps_threshold)
    lives = info['lives']

    new_state, reward, terminated, truncated, _ = env.step(action)

    # frame = env.render()
    # cv2.imshow("Breakout", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    episode_reward += reward

    if terminated or truncated:
        env.reset(seed=0)
        episode_reward_list.append(episode_reward)
        episode_reward = torch.tensor(0, dtype=torch.float64)

    replay_buffer.update(state, action, new_state, reward)

    state = new_state

    if len(replay_buffer) > BATCH_SIZE:
        loss = optimize_model(replay_buffer, policy_network, target_network, optimizer, criterion)

    if step % UPDATE_TARGET_NETWORK_STEPS == 0:
        target_net_state_dict = target_network.state_dict()
        policy_net_state_dict = policy_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_network.load_state_dict(target_net_state_dict)

    if len(episode_reward_list) > NUM_EPISODES_TO_AVERAGE:
        mean_episode_reward = torch.mean(torch.tensor(episode_reward_list))
        episode_reward_list = []
        print(f"Step: {step}, mean episode reward: {mean_episode_reward}, eps_threshold: {eps_threshold}, loss: {loss}")

    if step % 100_000 == 0:
        torch.save(policy_network.state_dict(), "dqn_breakout_ckpt.pth")

torch.save(policy_network.state_dict(), "dqn_breakout_trained_2.pth")


