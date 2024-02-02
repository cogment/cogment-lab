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
from typing import Any, TypedDict

import numpy as np
from cogment.environment import EnvironmentSession
from cogment.session import RecvEvent
from pettingzoo import AECEnv, ParallelEnv

from cogment_lab.core import CogmentEnv, State
from cogment_lab.generated.data_pb2 import Observation as PbObservation  # type: ignore
from cogment_lab.session_helpers import EnvironmentSessionHelper
from cogment_lab.specs import AgentSpecs
from cogment_lab.utils import import_object


class PZConfig(TypedDict):
    """Configuration for a PettingZoo environment."""

    env_path: str
    make_args: dict
    reset_options: dict


class AECEnvironment(CogmentEnv):
    """Cogment environment wrapper for PettingZoo AEC environments."""

    def __init__(
        self,
        env_path: str,
        make_kwargs: dict | None = None,
        reset_options: dict | None = None,
        render: bool = False,
        reinitialize: bool = False,
        dry: bool = False,
        sub_dry: bool = True,
    ):
        """
        Initialize the AECEnvironment.

        Args:
            env_path: Path to the PettingZoo environment class.
            make_kwargs: Arguments to pass to the environment constructor.
            reset_options: Options to pass to reset().
            render: Whether to render the environment.
            reinitialize: Whether to reinitialize the environment each session.
            dry: Whether to abstain from initializing the environment in this process.
            sub_dry: Whether to abstain from initializing the environment in the initializer of the subprocess
        """
        super().__init__(
            env_path=env_path,
            make_kwargs=make_kwargs,
            reset_options=reset_options,
            render=render,
            reinitialize=reinitialize,
            dry=sub_dry,
            sub_dry=sub_dry,
        )
        self.env_path = env_path
        self.make_args = make_kwargs or {}
        self.reset_options = reset_options or {}
        self.render = render
        self.reinitialize = reinitialize
        self.dry = dry

        logging.info(
            f"Creating AECEnvironment with {env_path=}, {make_kwargs=}, {reset_options=}, {render=}, {reinitialize=}, {dry=}, {sub_dry=}"
        )

        self.env_maker = import_object(self.env_path)

        assert callable(self.env_maker), f"Environment class at {self.env_path} is not callable"

        if render:
            self.make_args["render_mode"] = "rgb_array"

        if not self.dry:
            self.env: AECEnv = self.env_maker(**self.make_args)
            self.agent_specs = self.create_agent_specs(self.env)
        else:
            self.env = AECEnv()
            self.agent_specs = {}

        self.initialized = False

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
            self.env = self.env_maker(**self.make_args)
            self.agent_specs = self.create_agent_specs(self.env)

            state.env = self.env
            state.agent_specs = self.agent_specs
        elif self.initialized and not self.reinitialize:
            state.env = self.env
            state.agent_specs = self.agent_specs
        elif self.reinitialize:
            state.env = self.env_maker(**self.make_args)
            state.agent_specs = self.create_agent_specs(state.env)

        self.initialized = True

        state.environment_session = environment_session
        state.session_helper = EnvironmentSessionHelper(environment_session, state.agent_specs)
        state.session_cfg = state.environment_session.config
        state.actors = state.session_helper.actors

        state.observation_spaces = {agent: state.session_helper.get_observation_space(agent) for agent in state.actors}

        return state

    async def reset(self, state: State):
        """
        Reset the environment.

        Args:
            state: The Cogment state.

        Returns:
            A tuple of (state, observations)
        """
        logging.info("Resetting environment")

        state.env.reset(seed=state.session_cfg.seed)

        obs, reward, term, trunc, info = state.env.last()
        agent = state.env.agent_selection

        state.actor_name = agent

        frame = state.env.render() if state.session_cfg.render else None
        observation = state.observation_spaces[agent].create_serialize(
            value=obs, rendered_frame=frame, active=True, alive=True
        )
        if frame is not None:
            logging.info(f"Frame shape at reset: {frame.shape}")
        else:
            logging.info("Frame at reset is None")

        observations = {agent: observation}
        observations = self.fill_observations_(state=state, observations=observations, frame=frame)

        return state, observations

    async def step(self, state: State, action: Any):
        """
        Take an environment step.

        Args:
            state: The Cogment state.
            action: The action to take.

        Returns:
            A tuple of (state, observations, rewards, terminated, truncated, info)
        """
        logging.info("Stepping environment")

        state.env.step(action)
        obs, reward, terminated, truncated, info = state.env.last()
        agent = state.env.agent_selection

        frame = state.env.render() if state.session_cfg.render else None

        if frame is not None:
            logging.info(f"Frame shape at step: {frame.shape}")
        else:
            logging.info("Frame at step is None")

        observation = state.observation_spaces[agent].create_serialize(
            value=obs,
            rendered_frame=frame,
            active=True,
            alive=not (terminated or truncated),
        )

        # observations = [(agent, observation)]
        observations = {agent: observation}
        rewards = {agent: reward}
        terminateds = state.env.terminations
        truncateds = state.env.truncations

        state.actor_name = agent

        observations = self.fill_observations_(state, observations, frame=frame)

        return state, observations, rewards, terminateds, truncateds, info

    async def end(self, state: State):
        """
        End the environment session.

        Args:
            state: The Cogment state.
        """
        logging.info("Ending environment")
        state.env.close()

    @staticmethod
    def fill_observations_(
        state: State, observations: dict[str, PbObservation], frame: np.ndarray | None
    ) -> dict[str, PbObservation]:
        """
        Fill in any missing observations with the default observation. Mutates the observations dict.

        Args:
            state: The Cogment state.
            observations: The observations dict.
            frame: The rendered frame.

        Returns:
            The filled observations dict.
        """
        if "*" in observations:
            return observations
        for actor_name in state.actors:
            if actor_name not in observations:
                observations[actor_name] = state.observation_spaces[actor_name].create_serialize(
                    rendered_frame=frame, active=False
                )

        return observations

    @staticmethod
    def create_agent_specs(env: AECEnv):
        """
        Create the agent specs from a PettingZoo AEC environment.

        Args:
            env: The PettingZoo AEC environment.

        Returns:
            The agent specs dict.
        """
        # Check all observation and action spaces

        is_homogeneous = True
        observation_space = env.observation_space(env.possible_agents[0])
        action_space = env.action_space(env.possible_agents[0])
        for agent in env.possible_agents:
            if env.observation_space(agent) != observation_space:
                is_homogeneous = False
                break
            if env.action_space(agent) != action_space:
                is_homogeneous = False
                break

        if is_homogeneous:
            one_agent_specs = AgentSpecs.create_homogeneous(
                observation_space=observation_space,
                action_space=action_space,
            )
            agent_specs = {agent: one_agent_specs for agent in env.possible_agents}
        else:
            agent_specs = {
                agent: AgentSpecs.create_homogeneous(
                    observation_space=env.observation_space(agent),
                    action_space=env.action_space(agent),
                )
                for agent in env.possible_agents
            }

        return agent_specs


class ParallelEnvironment(CogmentEnv):
    """Cogment environment wrapper for PettingZoo Parallel environments."""

    def __init__(
        self,
        env_path: str,
        make_kwargs: dict | None = None,
        reset_options: dict | None = None,
        render: bool = False,
        reinitialize: bool = False,
        dry: bool = False,
        sub_dry: bool = True,
    ):
        """
        Initialize the ParallelEnvironment.

        Args:
            env_path: Path to the PettingZoo environment class.
            make_kwargs: Arguments to pass to the environment constructor.
            reset_options: Options to pass to reset().
            render: Whether to render the environment.
            reinitialize: Whether to reinitialize the environment each session.
            dry: Whether to abstain from initializing the environment in this process.
            sub_dry: Whether to abstain from initializing the environment in the initializer of the subprocess
        """
        super().__init__(
            env_path=env_path,
            make_kwargs=make_kwargs,
            reset_options=reset_options,
            render=render,
            reinitialize=reinitialize,
            dry=sub_dry,
            sub_dry=sub_dry,
        )
        self.env_path = env_path
        self.make_args = make_kwargs or {}
        self.reset_options = reset_options or {}
        self.render = render
        self.reinitialize = reinitialize
        self.dry = dry

        logging.info(
            f"Creating ParallelEnvironment with {env_path=}, {make_kwargs=}, {reset_options=}, {render=}, {reinitialize=}, {dry=}, {sub_dry=}"
        )

        self.env_maker = import_object(self.env_path)

        assert callable(self.env_maker), f"Environment class at {self.env_path} is not callable"

        if render:
            self.make_args["render_mode"] = "rgb_array"

        if not self.dry:
            self.env: ParallelEnv = self.env_maker(**self.make_args)
            self.agent_specs = self.create_agent_specs(self.env)
        else:
            self.env = ParallelEnv()
            self.agent_specs = {}

        self.initialized = False

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
            self.env: ParallelEnv = self.env_maker(**self.make_args)
            self.agent_specs = self.create_agent_specs(self.env)

            state.env = self.env
            state.agent_specs = self.agent_specs
        elif self.initialized and not self.reinitialize:
            state.env = self.env  # type: ignore
            state.agent_specs = self.agent_specs
        elif self.reinitialize:
            state.env = self.env_maker(**self.make_args)  # type: ignore
            state.agent_specs = self.create_agent_specs(state.env)

        self.initialized = True

        state.environment_session = environment_session
        state.session_helper = EnvironmentSessionHelper(environment_session, state.agent_specs)
        state.session_cfg = state.environment_session.config
        state.actors = state.session_helper.actors

        state.observation_spaces = {agent: state.session_helper.get_observation_space(agent) for agent in state.actors}

        return state

    async def reset(self, state: State):
        """
        Reset the environment.

        Args:
            state: The Cogment state.

        Returns:
            A tuple of (state, observations)
        """
        logging.info("Resetting environment")

        obs, info = state.env.reset(seed=state.session_cfg.seed)

        frame = state.env.render() if state.session_cfg.render else None

        if frame is not None:
            logging.info(f"Frame shape at reset: {frame.shape}")
        else:
            logging.info("Frame at reset is None")

        observations = {
            agent: state.observation_spaces[agent].create_serialize(
                value=obs[agent], rendered_frame=frame, active=True, alive=True
            )
            for agent in obs
        }

        # observations = {agent: observation}
        # observations = self.fill_observations_(state=state, observations=observations, frame=frame)

        return state, observations

    async def step(self, state: State, action: dict[str, Any]):
        """
        Take an environment step.

        Args:
            state: The Cogment state.
            action: The action to take.

        Returns:
            A tuple of (state, observations, rewards, terminated, truncated, info)
        """
        logging.info("Stepping environment")

        obs, rewards, terminated, truncated, info = state.env.step(action)

        frame = state.env.render() if state.session_cfg.render else None

        if frame is not None:
            logging.info(f"Frame shape at step: {frame.shape}")
        else:
            logging.info("Frame at step is None")

        observations = {
            agent: state.observation_spaces[agent].create_serialize(
                value=obs[agent],
                rendered_frame=frame,
                active=True,
                alive=not (terminated[agent] or truncated[agent]),
            )
            for agent in obs
        }

        return state, observations, rewards, terminated, truncated, info

    async def end(self, state: State):
        """
        End the environment session.

        Args:
            state: The Cogment state.
        """
        logging.info("Ending environment")
        state.env.close()

    async def read_actions(self, state: State, event: RecvEvent):
        """Read actions from event."""
        player_actions = {agent: state.session_helper.get_action(event, agent).value for agent in state.actors}
        return player_actions

    @staticmethod
    def create_agent_specs(env: ParallelEnv):
        """
        Create the agent specs from a PettingZoo AEC environment.

        Args:
            env: The PettingZoo AEC environment.

        Returns:
            The agent specs dict.
        """
        # Check all observation and action spaces

        is_homogeneous = True
        observation_space = env.observation_space(env.possible_agents[0])
        action_space = env.action_space(env.possible_agents[0])
        for agent in env.possible_agents:
            if env.observation_space(agent) != observation_space:
                is_homogeneous = False
                break
            if env.action_space(agent) != action_space:
                is_homogeneous = False
                break

        if is_homogeneous:
            one_agent_specs = AgentSpecs.create_homogeneous(
                observation_space=observation_space,
                action_space=action_space,
            )
            agent_specs = {agent: one_agent_specs for agent in env.possible_agents}
        else:
            agent_specs = {
                agent: AgentSpecs.create_homogeneous(
                    observation_space=env.observation_space(agent),
                    action_space=env.action_space(agent),
                )
                for agent in env.possible_agents
            }

        return agent_specs
