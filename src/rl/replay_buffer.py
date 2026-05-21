from __future__ import annotations

import random
from collections import deque
from typing import Deque, NamedTuple


class Transition(NamedTuple):
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.buffer: Deque[Transition] = deque(maxlen=int(capacity))

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(Transition(state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.buffer):
            raise ValueError("Cannot sample more items than the buffer contains")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
