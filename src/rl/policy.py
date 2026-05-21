from __future__ import annotations

import torch.nn as nn

class DQNPolicy(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        second_hidden = max(hidden_dim // 2, num_actions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, num_actions),
        )

    def forward(self, state):
        return self.net(state)
