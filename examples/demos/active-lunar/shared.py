# Copyright 2024 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib

import numpy as np
import torch
from torch import nn

from cogment_lab import Cogment


EPS_INIT = 0.9
EPS_FINAL = 0.001
EPS_DECAY = 300


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, seed: int = 0):
        self.capacity = capacity
        self.buffer_counter = 0

        self.rng = np.random.default_rng(seed)

        # Pre-allocate memory
        self.states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        index = self.buffer_counter % self.capacity

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.buffer_counter += 1

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        max_buffer_size = min(self.buffer_counter, self.capacity)
        batch_indices = self.rng.choice(max_buffer_size, batch_size, replace=False)

        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.dones[batch_indices],
        )

    def __len__(self):
        return min(self.buffer_counter, self.capacity)


def get_current_eps(
    current_step: int,
    eps_start: float = EPS_INIT,
    eps_final: float = EPS_FINAL,
    eps_decay_duration: int = EPS_DECAY,
) -> float:
    """
    Calculate the epsilon value for epsilon-greedy exploration in DQN.

    Parameters:
        current_step (int): The current step index.
        eps_start (float): The starting value of epsilon.
        eps_final (float): The final value of epsilon.
        eps_decay_duration (int): The number of steps over which epsilon is decayed linearly.

    Returns:
        float: The current epsilon value.
    """
    current_step = min(current_step, eps_decay_duration)

    decay_rate = (eps_start - eps_final) / eps_decay_duration

    current_epsilon = eps_start - decay_rate * current_step

    current_epsilon = max(current_epsilon, eps_final)

    return current_epsilon


def hash_model(model: nn.Module):
    model_state = model.state_dict()
    model_weights = []

    for key, value in model_state.items():
        model_weights.append(value.cpu().numpy().tobytes())

    model_hash = hashlib.sha256(b"".join(model_weights)).hexdigest()

    return model_hash


def dqn_loss(model: nn.Module, batch: tuple[np.ndarray, ...], γ: float) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch

    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions).to(torch.int64)
    rewards = torch.from_numpy(rewards)
    next_states = torch.from_numpy(next_states)
    dones = torch.from_numpy(dones)

    current_q_values = model(states)[0].gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states)[0].max(1)[0]
    expected_q_values = rewards + γ * next_q_values * (1 - dones)

    loss = nn.MSELoss()(current_q_values, expected_q_values.detach())

    return loss


async def evaluate_model(
    cog: Cogment, env_name: str, actor_impls: dict[str, str], num_episodes: int = 10
) -> tuple[float, float]:
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        trial_id = await cog.start_trial(
            env_name=env_name,
            actor_impls=actor_impls,
            session_config={"render": False, "seed": 10_000 + ep},
        )

        trial_data = await cog.get_trial_data(trial_id)
        dqn_data = trial_data["gym"]

        total_rewards.append(dqn_data.rewards.sum())
        episode_lengths.append(len(dqn_data.rewards))

    mean_reward = np.mean(total_rewards)
    mean_length = np.mean(episode_lengths)

    return mean_reward, mean_length
