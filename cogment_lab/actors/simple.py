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

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from cogment_lab.core import CogmentActor


class RandomActor(CogmentActor):
    def __init__(self, action_space: gym.spaces.Space):
        super().__init__(action_space)
        self.gym_action_space = action_space

    async def act(self, observation: Any, rendered_frame=None):
        return self.gym_action_space.sample()


class ConstantActor(CogmentActor):
    def __init__(self, action: Any):
        super().__init__(action)
        if isinstance(action, list):
            action = np.array(action)
        self.action = action

    async def act(self, observation: Any, rendered_frame=None):
        return self.action
