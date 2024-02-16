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

import gymnasium as gym

from cogment_lab.constants import DEFAULT_RENDERED_WIDTH
from cogment_lab.generated.data_pb2 import AgentSpecs as PbAgentSpecs  # type: ignore
from cogment_lab.specs import (
    ActionSpace,
    ObservationSpace,
    SerializationFormat,
    deserialize_space,
    serialize_gym_space,
)


class AgentSpecs:
    """
    Representation of the specification of an agent within Cogment Lab.
    """

    def __init__(self, agent_specs_pb: PbAgentSpecs):
        """
        AgentSpecs constructor.
        Shouldn't be called directly, prefer the factory function such as AgentSpecs.deserialize or AgentSpecs.create_homogeneous.
        """
        self._pb = agent_specs_pb

    def get_observation_space(self, render_width: int = DEFAULT_RENDERED_WIDTH) -> ObservationSpace:
        """
        Build an instance of the observation space for this agent

        Parameters:
            render_width: optional
                maximum width for the serialized rendered frame in observation

        NOTE: In the future we'll want to support different observation space per agent role
        """
        return ObservationSpace(deserialize_space(self._pb.observation_space), render_width)

    def get_action_space(self, seed: int | None = None) -> ActionSpace:
        """
        Build an instance of the action space for this agent

        Parameters:
            seed: optional
                the seed used when generating random actions

        NOTE: In the future we'll want to support different action space per agent roles
        """
        return ActionSpace(deserialize_space(self._pb.action_space), seed)

    @classmethod
    def create_homogeneous(
        cls,
        observation_space: gym.Space,
        action_space: gym.Space,
        serialization_format: SerializationFormat = SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an AgentSpecs.
        """
        return cls.deserialize(
            PbAgentSpecs(
                observation_space=serialize_gym_space(observation_space, serialization_format),
                action_space=serialize_gym_space(action_space, serialization_format),
            )
        )

    def serialize(self):
        """
        Serialize to a AgentSpecs protobuf message
        """
        return self._pb

    @classmethod
    def deserialize(cls, agent_specs_pb: PbAgentSpecs):
        """
        Factory function building an AgentSpecs instance from a AgentSpecs protobuf message
        """
        return cls(agent_specs_pb)
