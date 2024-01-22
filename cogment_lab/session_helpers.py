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

from cogment.actor import ActorSession
from cogment.environment import EnvironmentSession
from cogment.model_registry_v2 import ModelRegistry
from cogment.session import ActorInfo, RecvEvent

from cogment_lab.specs import AgentSpecs
from cogment_lab.specs.action_space import Action, ActionSpace
from cogment_lab.specs.observation_space import Observation, ObservationSpace


class ActorSessionHelper:
    """
    Cogment Verse actor session helper

    Provides additional methods to the regular Cogment actor session.
    """

    def __init__(self, actor_session: ActorSession, model_registry: ModelRegistry | None):
        self.actor_session = actor_session
        self.agent_specs = AgentSpecs.deserialize(self.actor_session.config.agent_specs)
        self.action_space = self.agent_specs.get_action_space(seed=self.actor_session.config.seed)
        self.observation_space = self.agent_specs.get_observation_space()
        self.model_registry = model_registry

    def get_action_space(self) -> ActionSpace:
        return self.action_space

    def get_observation_space(self) -> ObservationSpace:
        return self.observation_space

    def get_observation(self, event: RecvEvent) -> Observation | None:
        """
        Return the cogment verse observation for the current event.

        If the event does not contain an observation, return None.
        """
        if not event.observation:
            return None

        return self.observation_space.deserialize(event.observation.observation)

    def get_render(self, event: RecvEvent) -> bytes | None:
        """
        Return the render for the current event.

        If the event does not contain a render, return None.
        """
        if not event.observation:
            return None

        return event.observation.render


class EnvironmentSessionHelper:
    """
    A session helper for environments.
    """

    actor_infos: list[ActorInfo]

    def __init__(
        self,
        environment_session: EnvironmentSession,
        agent_specs: dict[str, AgentSpecs],
    ):
        self.actor_infos = environment_session.get_active_actors()

        assert set(agent_specs.keys()) == {
            actor_info.actor_name for actor_info in self.actor_infos
        }, f"Agent specs and active actors do not match. {agent_specs.keys()} != {self.actor_infos}"

        # Mapping actor_name to actor_idx
        self.actor_idxs = {actor_info.actor_name: actor_idx for (actor_idx, actor_info) in enumerate(self.actor_infos)}
        # Mapping actor_idx to actor_info
        self.actors = [actor_info.actor_name for actor_info in self.actor_infos]

        if isinstance(agent_specs, AgentSpecs):
            agent_specs = {actor_name: agent_specs for actor_name in self.actors}

        self.agent_specs = agent_specs
        self.observation_spaces = {
            agent_id: specs.get_observation_space(environment_session.config.render_width)
            for (agent_id, specs) in agent_specs.items()
        }

    def get_observation_space(self, actor_name: str) -> ObservationSpace:
        return self.observation_spaces[actor_name]

    def get_action_space(self, actor_name: str) -> ActionSpace:
        return self.agent_specs[actor_name].get_action_space()

    def _get_actor_idx(self, actor_name: str) -> int:
        actor_idx = self.actor_idxs.get(actor_name)

        if actor_idx is None:
            raise RuntimeError(f"No actor with name [{actor_name}] found!")

        return actor_idx

    def get_action(self, tick_data: Any, actor_name: str) -> Action | None:
        # For environments, tick_datas are events
        event: RecvEvent = tick_data

        if not event.actions or not event.actions:
            return None

        actor_idx = self._get_actor_idx(actor_name)
        action_space = self.get_action_space(actor_name)

        return action_space.deserialize(
            event.actions[actor_idx].action,
        )

    def get_observation(self, tick_data: Any, actor_name: str):
        """
        Return the cogment verse observation of a given actor at a tick.

        If no observation, returns None.
        """
        raise NotImplementedError

    def get_player_actions(self, tick_data: Any, actor_name: str) -> Action | None:
        """
        Return the cogment verse player action of a given actor at a tick.

        If only a single player actor is present, no `actor_name` is required.

        If no action, returns None.
        """
        event = tick_data
        if not event.actions:
            return None

        actions = [
            self.get_action(actor_name, tick_data)
            for player_actor_name in self.actors
            if player_actor_name == actor_name
        ]
        if len(actions) == 0:
            raise RuntimeError(f"No player actors having name [{actor_name}]")
        return actions[0]

    def get_player_observations(self, tick_data: Any, actor_name: str):
        if actor_name is None:
            observations = [self.get_observation(tick_data, actor_name) for player_actor_name in self.actors]
            if len(observations) == 0:
                raise RuntimeError("No player actors")
            if len(observations) > 1:
                raise RuntimeError("More than 1 player actor, please provide an actor name")
            return observations[0]

        observations = [
            self.get_observation(tick_data, actor_name)
            for player_actor_name in self.actors
            if player_actor_name == actor_name
        ]
        if len(observations) == 0:
            raise RuntimeError(f"No player actors having name [{actor_name}]")
        return observations[0]
