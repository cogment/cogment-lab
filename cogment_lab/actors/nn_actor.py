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

import numpy as np
import torch
from coltra import CAgent, DAgent, Observation
from coltra.models import BaseModel
from torch import nn
from torch.nn import functional as F

from cogment_lab.core import CogmentActor


class ColtraActor(CogmentActor):
    def __init__(self, model: BaseModel):
        super().__init__(model)
        self.model = model
        self.agent = DAgent(self.model) if self.model.discrete else CAgent(self.model)

    async def act(self, observation: np.ndarray, rendered_frame=None):
        obs = Observation(vector=observation)
        action, _, _ = self.agent.act(obs)
        return action.discrete


class NNActor(CogmentActor):
    def __init__(self, network: nn.Module, device: str = "cpu"):
        super().__init__(network=network)
        self.network = network
        self.device = device
        self.eps = 0.0
        self.num_actions: int | None = None
        self.rng = np.random.default_rng(0)

    async def act(self, observation: np.ndarray, rendered_frame=None) -> int:
        if self.num_actions is None:
            observation = observation.copy()
            obs = torch.from_numpy(observation).float().to(self.device)
            [act_probs] = self.network(obs)
            self.num_actions = act_probs.shape[0]

        if self.eps > 0.0 and self.rng.random() < self.eps:
            return self.rng.integers(0, self.num_actions)

        observation = observation.copy()
        obs = torch.from_numpy(observation).float().to(self.device)
        [act_probs] = self.network(obs)
        return act_probs.detach().cpu().numpy().argmax()

    def set_eps(self, eps: float):
        self.eps = eps


class BoltzmannActor(CogmentActor):
    def __init__(self, network: nn.Module, device: str = "cpu"):
        super().__init__(network=network)
        self.network = network
        self.device = device
        self.temperature = 1.0
        self.num_actions: int | None = None
        self.rng = np.random.default_rng(0)

    async def act(self, observation: np.ndarray, rendered_frame=None) -> int:
        observation = observation.copy()
        obs = torch.from_numpy(observation).float().to(self.device)
        with torch.no_grad():
            [act_vals] = self.network(obs)
            act_probs = F.softmax(act_vals / self.temperature, dim=0)

            action = torch.multinomial(act_probs, 1).item()

        return action

    def set_temperature(self, temperature: float):
        self.temperature = temperature
