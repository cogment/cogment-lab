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

import importlib
import logging
import os
from typing import Any, Callable

import gymnasium as gym
from cogment.environment import EnvironmentSession

from cogment_lab.core import CogmentEnv, State
from cogment_lab.session_helpers import EnvironmentSessionHelper
from cogment_lab.specs import AgentSpecs

log = logging.getLogger(__name__)

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GymEnvironment(CogmentEnv):
    """
    Gymnasium integration for Cogment.

    Exposes a Gymnasium environment as a Cogment environment.
    """

    session_helper: EnvironmentSessionHelper
    actor_name = "gym"

    def __init__(
        self,
        env_id: str | Callable[..., gym.Env],
        registration: str | None = None,
        make_kwargs: dict[str, Any] | None = None,
        reset_options: dict[str, Any] | None = None,
        render: bool = False,
        reinitialize: bool = False,
        dry: bool = False,
        sub_dry: bool = True,
    ):
        """
        Initialize the GymEnvironment.

        Args:
            env_id: The Gym environment ID.
            registration: Optional Gym registration string.
            make_kwargs: Optional args to pass to gym.make().
            reset_options: Optional reset options to pass to env.reset().
            render: Whether to render the environment.
            reinitialize: Whether to reinitialize the environment each session.
            dry: Whether to abstain from initializing the environment in this process.
            sub_dry: Whether to abstain from initializing the environment in the initializer of the subprocess.
        """
        super().__init__(
            env_id=env_id,
            registration=registration,
            make_kwargs=make_kwargs,
            reset_options=reset_options,
            render=render,
            reinitialize=reinitialize,
            dry=sub_dry,
            sub_dry=sub_dry,
        )

        self.env_id = env_id
        self.registration = registration
        self.make_kwargs = make_kwargs or {}
        self.reset_options = reset_options or {}
        self.render = render
        self.reinitialize = reinitialize
        self.dry = dry

        if "render_mode" in self.make_kwargs:
            raise ValueError("render_mode cannot be set in make_kwargs")

        if self.render:
            self.make_kwargs["render_mode"] = "rgb_array"

        if isinstance(self.env_id, Callable):
            self.env_maker = self.env_id
        else:
            self.env_maker = lambda **kwargs: gym.make(self.env_id, **kwargs)

        if self.registration:
            importlib.import_module(self.registration)

        if not self.dry:
            self.env = self.env_maker(**self.make_kwargs)

            self.agent_specs = {
                "gym": AgentSpecs.create_homogeneous(
                    observation_space=self.env.observation_space,
                    action_space=self.env.action_space,
                )
            }

            self.initialized = True
        else:
            self.env = None
            self.agent_specs = {}
            self.initialized = False

        self._is_closed = False

    def get_implementation_name(self):
        """
        Get the name of the Gym environment.

        Returns:
            The Gym environment ID.
        """
        return self.env_id

    def get_agent_specs(self):
        """
        Get the agent specs.

        Returns:
            The agent specs dict.
        """
        return self.agent_specs

    async def initialize(self, state: State, environment_session: EnvironmentSession):
        """
        Initialize the environment session.

        Args:
            state: The Cogment state.
            environment_session: The Cogment environment session.

        Returns:
            The updated state.
        """

        logging.info("Initializing environment session")

        if not self.initialized and not self.reinitialize:
            self.env = self.env_maker(**self.make_kwargs)
            self.agent_specs = {
                "gym": AgentSpecs.create_homogeneous(
                    observation_space=self.env.observation_space,
                    action_space=self.env.action_space,
                )
            }

            state.env = self.env
            state.agent_specs = self.agent_specs
        elif self.initialized and not self.reinitialize:
            state.env = self.env
            state.agent_specs = self.agent_specs
        elif self.reinitialize:
            state.env = self.env_maker(**self.make_kwargs)
            state.agent_specs = {
                "gym": AgentSpecs.create_homogeneous(
                    observation_space=state.env.observation_space,
                    action_space=state.env.action_space,
                )
            }

        state.environment_session = environment_session
        state.session_helper = EnvironmentSessionHelper(environment_session, state.agent_specs)
        state.session_cfg = state.environment_session.config
        state.actors = state.session_helper.actors
        state.actor_name = state.session_helper.actors[0]

        self.initialized = True

        return state

    async def reset(self, state: State):
        """
        Reset the environment.

        Args:
            state: The Cogment state.

        Returns:
            A tuple with the updated state and a dict of observations.
        """

        logging.info("Resetting environment")

        obs, _info = state.env.reset(seed=state.session_cfg.seed, options=state.session_cfg.reset_args)  # THIS

        state.observation_space = state.session_helper.get_observation_space(self.actor_name)
        frame = state.env.render() if state.session_cfg.render else None
        observation = state.observation_space.create_serialize(value=obs, rendered_frame=frame, active=True, alive=True)

        return state, {"*": observation}

    async def read_actions(self, state: State, event):
        """
        Read the agent action from the event.

        Args:
            state: The Cogment state.
            event: The event from Cogment.

        Returns:
            The agent action value.
        """
        player_action = state.session_helper.get_action(tick_data=event, actor_name=self.actor_name)
        return player_action.value

    async def step(self, state: State, action):
        """
        Step the environment.

        Args:
            state: The Cogment state.
            action: The agent action.

        Returns:
            A tuple with the updated state, observations dict, rewards dict,
            terminateds dict, truncateds dict, and info dict.
        """

        logging.info("Stepping environment")

        obs, reward, terminated, truncated, info = state.env.step(action)
        logging.info(f"Step returned {obs=}, {reward=}, {terminated=}, {truncated=}, {info=}")

        observation = state.observation_space.create_serialize(
            value=obs, rendered_frame=state.env.render() if state.session_cfg.render else None, active=True, alive=True
        )

        # observations = [("*", observation)]
        observations = {"*": observation}
        rewards = {self.actor_name: reward}
        terminateds = {self.actor_name: terminated}
        truncateds = {self.actor_name: truncated}

        return state, observations, rewards, terminateds, truncateds, info

    async def end(self, state: State):
        """
        End the environment session.

        Args:
            state: The Cogment state.

        Returns:
            The updated state.
        """

        logging.info("Ending environment")
        state.env.close()
        if self.env is not None and not self._is_closed:
            self.env.close()
        self._is_closed = True
