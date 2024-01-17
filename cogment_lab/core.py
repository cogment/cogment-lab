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

import abc
import copy
import logging
from typing import Awaitable, Callable, Generic, TypeVar

import cogment
import numpy as np
from cogment.actor import ActorSession
from cogment.environment import EnvironmentSession
from cogment.session import RecvEvent

from cogment_lab.session_helpers import ActorSessionHelper, EnvironmentSessionHelper
from cogment_lab.specs import (
    AgentSpecs
)

Action = TypeVar("Action")
Actions = dict[str, Action]

Observation = TypeVar("Observation")
Observations = dict[str, Observation]

Rewards = dict[str, float]

Dones = dict[str, bool]


class State:
    """Class to hold state information."""

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        """Get attribute from internal dictionary."""
        return self.__dict__.get(name, None)


class BaseEnv(abc.ABC):
    """Base environment class."""

    agent_specs: dict[str, AgentSpecs]

    def __init__(self, *args, **kwargs):
        """Initialize with arguments."""
        self.args = copy.deepcopy(args)
        self.kwargs = copy.deepcopy(kwargs)

    async def impl(self, environment_session: EnvironmentSession):
        """Abstract method to implement environment logic."""
        raise NotImplementedError()

    def get_constructor(self):
        """Get constructor for this environment."""
        cls = self.__class__
        return lambda: cls(*self.args, **self.kwargs)


class CogmentEnv(BaseEnv, abc.ABC, Generic[Observation, Action]):
    """Base Cogment environment class."""

    environment_session: EnvironmentSession
    session_helper: EnvironmentSessionHelper
    agent_specs: dict[str, AgentSpecs]
    actor_name: str

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

    async def initialize(self, state: State, environment_session: EnvironmentSession):
        """Initialize state and session."""
        state.environment_session = environment_session
        return state

    @abc.abstractmethod
    async def reset(self, state: State) -> tuple[State, Observations]:
        """Reset environment state.

        Returns:
            State: Updated state.
            Observations: Initial observations.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def step(self, state: State, action: Actions) -> tuple[State, Observations, Rewards, Dones, Dones, dict]:
        """Take a step in the environment.

        Args:
            state: Current state.
            action: Actions from actors.

        Returns:
            State: Updated state.
            Observations: New observations.
            Rewards: Rewards for each actor.
            Dones: Whether each actor is done.
            Dones: Whether each actor is truncated.
            dict: Additional info.
        """
        raise NotImplementedError()

    async def end(self, state: State):
        """Clean up when done."""
        pass

    async def read_actions(self, state: State, event: RecvEvent):
        """Read actions from event."""
        player_action = state.session_helper.get_action(event, state.actor_name)
        return player_action.value

    async def impl(self, environment_session: EnvironmentSession):
        """Implement environment logic."""
        state = State()
        state = await self.initialize(state, environment_session)
        state, observations = await self.reset(state)

        observations = list(observations.items())

        logging.info(f"Starting environment session")

        environment_session.start(observations)

        async for event in environment_session.all_events():
            event: RecvEvent
            if event.actions:
                actions = await self.read_actions(state, event)
                state, observations, rewards, terminateds, truncateds, info = await self.step(state, actions)

                dones = {actor_name: terminateds[actor_name] or truncateds[actor_name] for actor_name in terminateds}

                logging.info(f"Adding rewards: {rewards}")
                for actor_name in state.actors:
                    if actor_name not in rewards:
                        rewards[actor_name] = float("nan")
                for actor_name, reward in rewards.items():
                    environment_session.add_reward(value=reward, to=[actor_name], confidence=1.0)

                observations = list(observations.items())

                if all(dones.values()):
                    logging.info(f"Logging dones=True")
                    environment_session.end(observations)
                # elif event.type != cogment.EventType.ACTIVE:
                #     logging.info("Logging event.type!=ACTIVE")
                #     environment_session.end(observations)
                else:
                    logging.info(f"Logging a normal observation")
                    environment_session.produce_observations(observations)

        await self.end(state)


class BaseActor(abc.ABC):
    """Base actor class."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        self.args = args
        self.kwargs = kwargs

    async def impl(self, actor_session: ActorSession):
        """Abstract method to implement actor logic."""
        raise NotImplementedError()


class NativeActor(BaseActor):
    """Native actor wrapping a function."""

    def __init__(self, impl: Callable[[ActorSession], Awaitable]):
        """Initialize with implementation function."""
        super().__init__(impl)
        self._impl = impl

    async def impl(self, actor_session: ActorSession):
        """Call implementation function."""
        await self._impl(actor_session)


class CogmentActor(BaseActor, abc.ABC, Generic[Observation, Action]):
    """Base Cogment actor class."""

    actor_session: ActorSession
    current_event: RecvEvent
    session_helper: ActorSessionHelper

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

    async def initialize(self, actor_session: ActorSession):
        """Initialize session and helpers."""
        self.actor_session = actor_session
        self.actor_session.start()

        self.session_helper = ActorSessionHelper(actor_session, None)
        self.action_space = self.session_helper.get_action_space()

    @abc.abstractmethod
    async def act(self, observation: Observation, rendered_frame: np.ndarray | None = None) -> Action:
        """Choose an action based on observation.

        Args:
            observation: Current observation.
            rendered_frame: Optional rendered frame.

        Returns:
            Action to take.
        """
        raise NotImplementedError()

    async def on_reward(self, rewards: list):
        """Handle received rewards."""
        pass

    async def on_message(self, messages: list):
        """Handle received messages."""
        pass

    async def end(self):
        """Clean up when done."""
        pass

    async def impl(self, actor_session: ActorSession):
        """Implement actor logic."""
        await self.initialize(actor_session)
        async for event in actor_session.all_events():
            event: RecvEvent
            self.current_event = event
            if event.type != cogment.EventType.ACTIVE:
                logging.info(f"Skipping event of type {event.type}")
                continue

            if event.observation:
                observation = self.session_helper.get_observation(event)
                logging.info(f"Got observation: {observation}")

                if not observation.active:
                    action = None
                elif not observation.alive:
                    action = None
                else:
                    action = await self.act(observation.value, observation.rendered_frame)
                logging.info(f"Got action: {action} with action_space: {self.action_space.gym_space}")
                cog_action = self.action_space.create_serialize(action)
                actor_session.do_action(cog_action)
            if event.rewards:
                await self.on_reward(event.rewards)
            if event.messages:
                await self.on_message(event.messages)

        await self.end()
