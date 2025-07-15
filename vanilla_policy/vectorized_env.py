import torch
from torch import nn
import gymnasium as gym
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import sys
from time import time


BATCH_SIZE = 5000
LEARNING_RATE = 1e-2
NUM_EPOCHS = 50

torch.manual_seed(0)


def get_weights_from_rewards(rews: list[np.ndarray]) -> np.ndarray:
    rews = np.array(rews)
    
    for i in range(rews.shape[1]):
        ep_rews_splitted = np.split(rews[:, i].reshape(1, -1), np.argwhere(rews[:, i] == 0).flatten(), axis=1)

        weights = [ep_rew.sum() for ep_rew in ep_rews_splitted]

        final_weights = []

        for weight, ep_rew in zip(weights, ep_rews_splitted):
            final_weights += (ep_rew*weight).flatten().tolist()
        
        rews[:, i] = np.array(final_weights)

    return rews

def get_mean_episode_rews(rews: list[np.ndarray]) -> np.ndarray:
    rews = np.array(rews)

    all_weights = []
    
    for i in range(rews.shape[1]):
        ep_rews_splitted = np.split(rews[:, i].reshape(1, -1), np.argwhere(rews[:, i] == 0).flatten(), axis=1)

        weights = [ep_rew.sum() for ep_rew in ep_rews_splitted]

        all_weights += weights

    return np.array(all_weights).mean()

class Mlp(nn.Module):
    def __init__(self, observation_dim, actions_dim, sizes):
        super().__init__()
        input_layer = nn.Sequential(nn.Linear(observation_dim, sizes[0]), nn.ReLU())

        layers = []
        for i in range(1, len(sizes)):
            layer = nn.Sequential(nn.Linear(sizes[i-1], sizes[i]), nn.ReLU())
            layers.append(layer)

        output_layer = nn.Linear(sizes[-1], actions_dim)

        self.fc = nn.Sequential(input_layer, *layers, output_layer)

    def forward(self, x):
        return self.fc(x)

def get_policy(model, obs):
    logits = model(obs)
    return Categorical(logits=logits)

def get_action(model, obs):
    return get_policy(model, obs).sample()

def compute_loss(model, obs, action, weights):
    """
    Weights are in this case rewards
    """
    # Returns natural logarithm of a probability of selecting action at this index 
    log_prob = get_policy(model, obs).log_prob(action)
    return -(weights*log_prob).mean()

vec_env = gym.make_vec("CartPole-v1", num_envs=8, render_mode="rgb_array", vectorization_mode="sync")

observation_dim = vec_env.observation_space.shape[1]

actions_dim = vec_env.action_space[0].n

model = Mlp(observation_dim, actions_dim, [32])

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter("runs/")

start_time = time()

for epoch in range(NUM_EPOCHS):
    batch_obs = [] # List for storing all observations
    batch_acts = [] # List for storing all actions
    batch_lens = [] # List for storing batch lengths 
    batch_rets = [] # List for storing batch returns

    ep_rews = [] # List for storing eposiode rewards


    obs, info = vec_env.reset()
    while True:
        batch_obs.append(obs.tolist())

        actions = get_action(model, torch.tensor(obs))
        obs, reward, terminated, truncated, _ = vec_env.step(actions.numpy())

        batch_acts.append(actions.tolist())
        ep_rews.append(reward)

        if len(batch_obs) > BATCH_SIZE:
            break

    mean_batch_return = get_mean_episode_rews(ep_rews)

    batch_weights = get_weights_from_rewards(ep_rews)

    batch_weights = torch.tensor(batch_weights)
    batch_obs = torch.tensor(batch_obs)

    batch_acts = torch.tensor(batch_acts)

    optimizer.zero_grad()
    batch_loss = compute_loss(model, batch_obs, batch_acts, batch_weights)
    batch_loss.backward()
    optimizer.step()

    writer.add_scalar("Mean return", mean_batch_return, epoch)

    print(f"Epoch {epoch}: Mean return: {mean_batch_return}")

end_time = time()

print(f"Training took: {end_time - start_time}s")

# Inference
obs, info = vec_env.reset()
while True:
    frame = vec_env.render()[0]
    cv2.imshow("env 1", frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    
    actions = get_action(model, torch.tensor(obs))
    obs, reward, terminated, truncated, _ = vec_env.step(actions.numpy())
