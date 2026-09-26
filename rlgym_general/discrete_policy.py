import torch.nn as nn
import torch
import numpy as np


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)
        self.n_actions = n_actions

    def get_output(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        return self.model(obs)

    def get_action(self, obs, deterministic=True):
        probs = self.get_output(obs).view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)
        if deterministic:
            return probs.cpu().numpy().argmax(), 0
        action = torch.multinomial(probs, 1, True)
        return action.flatten().cpu(), torch.log(probs).gather(-1, action).flatten().cpu()