from __future__ import annotations

import random

import numpy as np

from src.rl.action import Action, enabled_action_values
from src.utils.bbox import box_center, center_inside, containment_ratio


def epsilon_by_step(start: float, end: float, step: int, decay: float) -> float:
    if decay <= 0:
        return float(end)
    if decay < 1.0:
        return float(max(end, start * (decay ** step)))
    ratio = min(max(step / decay, 0.0), 1.0)
    return float(start + ratio * (end - start))


def linear_decay(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    ratio = min(max(step / decay_steps, 0.0), 1.0)
    return float(start + ratio * (end - start))


def choose_action_towards_nearest_hard_object(env) -> Action:
    roi = env.current_roi
    roi_cx, roi_cy = box_center(roi)
    allowed_values = enabled_action_values(env.config)
    candidates = []
    for obj in env.hard_objects:
        obj_id = obj.get("id")
        if obj_id in env.covered_object_ids:
            continue
        box = obj.get("bbox_xyxy")
        if not box:
            continue
        cx, cy = box_center(box)
        candidates.append(((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2, box, cx, cy))

    if not candidates:
        allowed = [Action(value) for value in allowed_values if value != Action.STOP.value]
        return random.choice(allowed or [Action.STOP])

    _, box, cx, cy = min(candidates, key=lambda item: item[0])
    if center_inside(box, roi):
        if containment_ratio(box, roi) < float(env.config.get("min_coverage", 0.5)) and Action.ZOOM_OUT.value in allowed_values:
            return Action.ZOOM_OUT
        if Action.ZOOM_IN.value in allowed_values:
            return Action.ZOOM_IN
        move_values = {Action.MOVE_UP.value, Action.MOVE_DOWN.value, Action.MOVE_LEFT.value, Action.MOVE_RIGHT.value}
        moves = [Action(value) for value in allowed_values if value in move_values]
        return random.choice(moves or [Action.STOP])

    dx = cx - roi_cx
    dy = cy - roi_cy
    if abs(dx) > abs(dy):
        return Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT
    return Action.MOVE_DOWN if dy > 0 else Action.MOVE_UP


def select_action(state, policy, epsilon: float, guided_prob: float, env) -> int:
    allowed_values = enabled_action_values(env.config)
    if random.random() < epsilon:
        if bool(env.config.get("guided_exploration", True)) and random.random() < guided_prob:
            action = choose_action_towards_nearest_hard_object(env).value
            return action if action in allowed_values else random.choice(allowed_values)
        return random.choice(allowed_values)
    if policy is None:
        return Action.STOP.value

    import torch

    with torch.no_grad():
        tensor_state = _to_tensor(state)
        if tensor_state.dim() == 1:
            tensor_state = tensor_state.unsqueeze(0)
        tensor_state = tensor_state.to(next(policy.parameters()).device)
        q_values = policy(tensor_state)
        mask = torch.full_like(q_values, float("-inf"))
        mask[:, allowed_values] = 0.0
        return int((q_values + mask).argmax(dim=1).item())


def optimize_dqn(batch, policy, target_policy, optimizer, gamma: float, allowed_actions: list[int] | None = None):
    import torch
    import torch.nn.functional as F

    device = next(policy.parameters()).device
    states = torch.stack([_to_tensor(t.state) for t in batch]).to(device)
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.stack([_to_tensor(t.next_state) for t in batch]).to(device)
    dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=device)

    q_values = policy(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = target_policy(next_states)
        if allowed_actions is not None:
            mask = torch.full_like(next_q_values, float("-inf"))
            mask[:, [int(value) for value in allowed_actions]] = 0.0
            next_q_values = next_q_values + mask
        next_q = next_q_values.max(dim=1).values
        target_q = rewards + gamma * next_q * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _to_tensor(value):
    import torch

    if hasattr(value, "detach"):
        return value.detach().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)
