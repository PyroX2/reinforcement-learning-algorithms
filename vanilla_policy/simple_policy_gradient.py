import torch
from torch import nn
import gymnasium as gym
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from time import time


BATCH_SIZE = 5000
LEARNING_RATE = 1e-2
NUM_EPOCHS = 50

torch.manual_seed(0)


class Mlp(nn.Module):
    def __init__(self, obsevration_dim, actions_dim, sizes):
        super().__init__()
        input_layer = nn.Sequential(nn.Linear(obsevration_dim, sizes[0]), nn.ReLU())

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

def train_one_epoch(model, optimizer, env):
    batch_obs = [] # List for storing all observations
    batch_acts = [] # List for storing all actions
    batch_weights = [] # List for storing weights (episode rewards, episode reward is asigned as a weight to every step in an episode)
    batch_lens = [] # List for storing batch lengths 
    batch_rets = [] # List for storing batch returns

    ep_rews = [] # List for storing eposiode rewards

    obs, info = env.reset()
    while True:
        batch_obs.append(obs.tolist())
        # env.render()
        
        action = get_action(model, torch.tensor(obs))
        obs, reward, terminated, truncated, _ = env.step(action.item())

        batch_acts.append(action)
        ep_rews.append(reward)

        if terminated or truncated:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # Propagate total trajcetory reward so that it can be multiplied with every probability of state, action pair
            batch_weights += [ep_ret]*ep_len

            obs, info = env.reset()
            ep_rews = []

            if len(batch_obs) > BATCH_SIZE:
                break

    optimizer.zero_grad()
    batch_loss = compute_loss(model, torch.tensor(batch_obs), torch.tensor(batch_acts), torch.tensor(batch_weights))
    batch_loss.backward()
    optimizer.step()

    return batch_rets, batch_lens

def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    observation_dim = env.observation_space.shape[0]

    actions_dim = 2

    model = Mlp(observation_dim, actions_dim, [32])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter("runs/")

    start_time = time()

    for epoch in range(NUM_EPOCHS):
        batch_rets, batch_lens = train_one_epoch(model, optimizer, env)

        mean_batch_return = np.array(batch_rets).mean()
        mean_batch_lens = np.array(batch_lens).mean()

        writer.add_scalar("Mean return", mean_batch_return, epoch)
        writer.add_scalar("Mean episode len", mean_batch_lens, epoch)

        print(f"Epoch {epoch}: Mean return: {mean_batch_return}, Mean episode len: {mean_batch_lens}")

    end_time = time()

    print(f"Training took: {end_time - start_time}")
    


if __name__ == "__main__":
    main()

