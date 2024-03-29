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

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from cogment_lab.core import Action, CogmentActor, Observation


class EchoEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Text(max_length=10, min_length=1)
        self.action_space = gym.spaces.Text(max_length=10, min_length=1)
        self.text = ""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[str, dict]:
        text = "".join(self.np_random.choice(self.observation_space.character_list, size=10))
        self.text = text
        return text, {}

    def step(self, action: str):
        reward = float(action == self.text)
        truncated = False
        terminated = action == self.text
        return self.text, reward, terminated, truncated, {}


class NoisyEchoActor(CogmentActor):
    def __init__(self, noise_prob: float = 0.1):
        super().__init__(noise_prob=noise_prob)
        self.noise_prob = noise_prob
        self.rng = np.random.default_rng(0)

    async def act(self, observation: Observation, rendered_frame: np.ndarray | None = None) -> Action:
        if self.rng.random() < self.noise_prob:
            return "nope"
        return observation


gym.register("Echo-v0", entry_point="tests.shared:EchoEnv", max_episode_steps=10)
