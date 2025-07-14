import torch
from torch import nn
import gymnasium as gym
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

NUM_EPOCHS = 100

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
    
# Get policy output
def get_policy(model, obs):
    logits = model(obs)
    return Categorical(logits=logits)

# Get action (get policy output and sample from it)
def get_action(model, obs):
    return get_policy(model, obs).sample()

def compute_loss(model, obs, action, weights):
    """
    Weights are in this case rewards
    """
    # Returns natural logarithm of a probability of selecting this action
    log_prob = get_policy(model, obs).log_prob(action)
    return -(weights*log_prob).mean() # Maximize the log probability of actions that lead to big rewards

def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    observation_dim = env.observation_space.shape[0]

    actions_dim = 4

    model = Mlp(observation_dim, actions_dim, [32, 32])
    model.load_state_dict(torch.load("/Users/jakub/machine_learning/rl/spinningup/vanilla_policy/lunar_lander_trained.pt", weights_only=True))

    for epoch in range(NUM_EPOCHS):
        obs, info = env.reset()
        while True:
            env.render()
            
            action = get_action(model, torch.tensor(obs))
            obs, reward, terminated, truncated, _ = env.step(action.item())

            if terminated or truncated:
                obs, info = env.reset()

if __name__ == "__main__":
    main()

