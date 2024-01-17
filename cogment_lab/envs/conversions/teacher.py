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


import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector


class GymTeacherAEC(AECEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        gym_env_name: str,
        gym_make_kwargs: dict = {},
        render_mode: str | None = None,
    ):
        super().__init__()
        self.gym_env = gym.make(gym_env_name, render_mode=render_mode, **gym_make_kwargs)
        self.possible_agents = ["gym", "teacher"]
        self.observation_spaces = {
            "gym": self.gym_env.observation_space,
            "teacher": gym.spaces.Dict(
                {
                    "observation": self.gym_env.observation_space,
                    "action": self.gym_env.action_space,
                }
            ),
        }
        teacher_action_space = gym.spaces.Dict({"active": gym.spaces.Discrete(2), "action": self.gym_env.action_space})
        self.action_spaces = {
            "gym": self.gym_env.action_space,
            "teacher": teacher_action_space,
        }
        self._agent_selector = agent_selector(self.possible_agents)
        self.override = False
        # self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._last_observation, info = self.gym_env.reset(seed=seed, options=options)
        self.infos = {"gym": info, "teacher": info}
        self._gym_action = None

    def step(self, action):
        current_agent = self.agent_selection
        next_agent = self._agent_selector.next()

        if current_agent == "gym":
            self._cumulative_rewards["gym"] = 0
            self._cumulative_rewards["teacher"] = 0

            self._gym_action = action

        elif current_agent == "teacher":
            if action["active"] == 1:
                self.override = True
                real_action = action["action"]
            else:
                self.override = False
                real_action = self._gym_action
            observation, reward, terminated, truncated, info = self.gym_env.step(real_action)
            self._last_observation = observation
            self.rewards["gym"] = float(reward)
            self.rewards["teacher"] = float(reward)

            self.terminations["gym"] = terminated
            self.terminations["teacher"] = terminated

            self.truncations["gym"] = truncated
            self.truncations["teacher"] = truncated

            self.infos["gym"] = info
            self.infos["teacher"] = info

            self._accumulate_rewards()
        self.agent_selection = next_agent

    def observe(self, agent):
        if agent == "gym":
            return self._last_observation
        elif agent == "teacher":
            return {"observation": self._last_observation, "action": self._gym_action}

    def render(self):
        img = self.gym_env.render()
        if self.override:
            assert isinstance(
                img, np.ndarray
            ), "Gym environment must return a numpy array when rendering. Make sure you passed render_mode='rgb_array' to the environment."
            W, H, _ = img.shape
            N = 5

            # Set the borders to red
            img[:N, :, :] = [255, 0, 0]  # Top border
            img[-N:, :, :] = [255, 0, 0]  # Bottom border
            img[:, :N, :] = [255, 0, 0]  # Left border
            img[:, -N:, :] = [255, 0, 0]  # Right border
        return img

    def close(self):
        self.gym_env.close()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]


class GymTeacherParallel(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        gym_env_name: str,
        gym_make_kwargs: dict = {},
        render_mode: str | None = None,
    ):
        super().__init__()
        self.gym_env = gym.make(gym_env_name, render_mode=render_mode, **gym_make_kwargs)
        self.possible_agents = ["gym", "teacher"]
        self.observation_spaces = {
            "gym": self.gym_env.observation_space,
            "teacher": self.gym_env.observation_space,
        }

        teacher_action_space = gym.spaces.Dict({"active": gym.spaces.Discrete(2), "action": self.gym_env.action_space})
        self.action_spaces = {
            "gym": self.gym_env.action_space,
            "teacher": teacher_action_space,
        }
        self._agent_selector = agent_selector(self.possible_agents)
        self.override = False
        # self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = self.possible_agents[:]

        obs, info = self.gym_env.reset(seed=seed, options=options)

        infos = {"gym": info, "teacher": info}
        observations = {"gym": obs, "teacher": obs}

        return observations, infos

    def step(self, action):
        if action["teacher"]["active"] == 1:
            self.override = True
            real_action = action["teacher"]["action"]
        else:
            self.override = False
            real_action = action["gym"]

        observation, reward, terminated, truncated, info = self.gym_env.step(real_action)

        observations = {"gym": observation, "teacher": observation}
        rewards = {"gym": reward, "teacher": reward}
        terminations = {"gym": terminated, "teacher": terminated}
        truncations = {"gym": truncated, "teacher": truncated}
        infos = {"gym": info, "teacher": info}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        img = self.gym_env.render()
        if self.override:
            assert isinstance(
                img, np.ndarray
            ), "Gym environment must return a numpy array when rendering. Make sure you passed render_mode='rgb_array' to the environment."
            W, H, _ = img.shape
            N = 5

            # Set the borders to red
            img[:N, :, :] = [255, 0, 0]  # Top border
            img[-N:, :, :] = [255, 0, 0]  # Bottom border
            img[:, :N, :] = [255, 0, 0]  # Left border
            img[:, -N:, :] = [255, 0, 0]  # Right border
        return img

    def close(self):
        self.gym_env.close()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
