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
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector


class GymObserverAEC(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "GymWithObserverEnv"}

    def __init__(
        self,
        gym_env_name: str,
        gym_make_kwargs: dict = {},
        render_mode: str | None = None,
    ):
        super().__init__()
        logging.info(
            f"Creating GymObserverAEC with gym_env_name={gym_env_name}, gym_make_kwargs={gym_make_kwargs}, render_mode={render_mode}"
        )
        self.gym_env = gym.make(gym_env_name, render_mode=render_mode, **gym_make_kwargs)
        self.possible_agents = ["gym", "observer"]
        self.observation_spaces = {
            "gym": self.gym_env.observation_space,
            "observer": gym.spaces.Dict(
                {
                    "observation": self.gym_env.observation_space,
                    "action": self.gym_env.action_space,
                }
            ),
        }
        self.action_spaces = {
            "gym": self.gym_env.action_space,
            "observer": self.gym_env.action_space,  # Even though actions are ignored
        }
        self._agent_selector = agent_selector(self.possible_agents)
        # self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        logging.info(f"Resetting GymObserverAEC with seed={seed}, options={options}")
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._last_observation, info = self.gym_env.reset()
        self.infos = {"gym": info, "observer": info}
        self._gym_action = None

    def step(self, action):
        logging.info(f"Stepping GymObserverAEC with action={action}")
        current_agent = self.agent_selection

        if current_agent == "gym":
            # Execute main agent's action directly
            observation, reward, terminated, truncated, info = self.gym_env.step(action)
            self._last_observation = observation
            self.rewards["gym"] = float(reward)
            self.terminations["gym"] = terminated
            self.truncations["gym"] = truncated
            self.infos["gym"] = info

            self.rewards["observer"] = self.rewards["gym"]
            self.terminations["observer"] = self.terminations["gym"]
            self.truncations["observer"] = self.truncations["gym"]
            self.infos["observer"] = self.infos["gym"]

            self._gym_action = action

        # Observer agent's turn; actions are ignored
        elif current_agent == "observer":
            pass

        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        if agent == "gym":
            return self._last_observation
        elif agent == "observer":
            return {"observation": self._last_observation, "action": self._gym_action}

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]


class GymObserverParallel(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "GymWithObserverEnv"}

    def __init__(
        self,
        gym_env_name: str,
        gym_make_kwargs: dict = {},
        render_mode: str | None = None,
    ):
        super().__init__()
        logging.info(
            f"Creating GymObserverParallel with gym_env_name={gym_env_name}, gym_make_kwargs={gym_make_kwargs}, render_mode={render_mode}"
        )
        self.gym_env = gym.make(gym_env_name, render_mode=render_mode, **gym_make_kwargs)
        self.possible_agents = ["gym", "observer"]
        self.observation_spaces = {
            "gym": self.gym_env.observation_space,
            "observer": self.gym_env.observation_space,
        }
        self.action_spaces = {
            "gym": self.gym_env.action_space,
            "observer": self.gym_env.action_space,  # Even though actions are ignored
        }
        # self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        logging.info(f"Resetting GymObserverParallel with seed={seed}, options={options}")
        self.agents = self.possible_agents[:]

        obs, info = self.gym_env.reset(seed=seed, options=options)

        infos = {"gym": info, "observer": info}
        observations = {"gym": obs, "observer": obs}

        return observations, infos

    def step(self, action: dict[str, Any]):
        logging.info(f"Stepping GymObserverParallel with action={action}")

        obs, reward, terminated, truncated, info = self.gym_env.step(action["gym"])

        observations = {"gym": obs, "observer": obs}
        rewards = {"gym": reward, "observer": reward}
        terminations = {"gym": terminated, "observer": terminated}
        truncations = {"gym": truncated, "observer": truncated}
        infos = {"gym": info, "observer": info}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
