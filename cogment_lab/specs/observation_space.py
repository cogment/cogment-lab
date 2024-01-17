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

from cogment_lab.constants import DEFAULT_RENDERED_WIDTH
from cogment_lab.generated.data_pb2 import Observation as PbObservation

from .encode_rendered_frame import decode_rendered_frame, encode_rendered_frame
from .ndarray_serialization import deserialize_ndarray, serialize_ndarray


# pylint: disable=attribute-defined-outside-init
class Observation:
    """
    Cogment Verse actor observation

    Properties:
        flat_value:
            The observation value, as a flat numpy array.
        value:
            The observation value, as a numpy array.
        active: optional
            Boolean indicating if the object is active.
        alive: optional
            Boolean indicating if the object is alive.
        rendered_frame: optional
            Environment's rendered frame as a numpy array of RGB pixels.
    """

    def __init__(
        self,
        gym_space: gym.Space,
        pb_observation=None,
        value=None,
        active=None,
        alive=None,
        rendered_frame=None,
    ):
        """
        Observation constructor.
        Shouldn't be called directly, prefer the factory function of ObservationSpace.
        """

        self._gym_space = gym_space

        if pb_observation is not None:
            assert value is None
            assert active is None
            assert alive is None
            assert rendered_frame is None
            self._pb_observation = pb_observation
            return

        self._value = value
        self._active = active
        self._alive = alive
        self._rendered_frame = rendered_frame

        self._pb_observation = PbObservation(
            active=active,
            alive=alive,
        )

    def _compute_flat_value(self):
        if hasattr(self, "_value"):
            return gym.spaces.flatten(self._gym_space, self._value)

        if not self._pb_observation.value != b"" or self._pb_observation.value is None:
            return None

        return deserialize_ndarray(self._pb_observation.value)

    @property
    def flat_value(self):
        if not hasattr(self, "_flat_value"):
            self._flat_value = self._compute_flat_value()
        return self._flat_value

    def _compute_value(self):
        return gym.spaces.unflatten(self._gym_space, self.flat_value) if self.flat_value is not None else None

    @property
    def value(self):
        if not hasattr(self, "_value"):
            self._value = self._compute_value()
        return self._value

    def _deserialize_rendered_frame(self):
        if not self._pb_observation.rendered_frame != b"":
            return None
        return decode_rendered_frame(self._pb_observation.rendered_frame)

    @property
    def rendered_frame(self):
        if not hasattr(self, "_rendered_frame"):
            self._rendered_frame = self._deserialize_rendered_frame()
        return self._rendered_frame

    @property
    def active(self):
        return self._pb_observation.active if self._pb_observation.active != b"" else self._active

    @property
    def alive(self):
        return self._pb_observation.alive if self._pb_observation.alive != b"" else self._alive

    def __repr__(self):
        return f"Observation(value={self.value.shape if isinstance(self.value, np.ndarray) else self.value}, active={self.active}, alive={self.alive}, rendered_frame={self.rendered_frame.shape if self.rendered_frame is not None else 'None'})@{hex(id(self))}"  # type: ignore

    def __str__(self):
        return self.__repr__()


class ObservationSpace:
    """
    Cogment Verse observation space

    Properties:
        gym_space:
            Wrapped Gym space for the observation values
        render_width:
            Maximum width for the serialized rendered frame in observations
    """

    def __init__(self, space: gym.Space, render_width: int = DEFAULT_RENDERED_WIDTH):
        """
        ObservationSpace constructor.
        Shouldn't be called directly, prefer the factory function of EnvironmentSpecs.
        """
        if isinstance(space, gym.spaces.Dict) and ("action_mask" in space.spaces):
            # Check the observation space defines an action_mask "component" (like PettingZoo does)
            assert "observation" in space.spaces
            assert len(space.spaces) == 2

            self.gym_space = space.spaces["observation"]
            self.action_mask_gym_space = space.spaces["action_mask"]
        else:
            # "Standard" observation space, no action_mask
            self.gym_space = space
            self.action_mask_gym_space = None

        # Other configuration
        self.render_width = render_width

    def create(
        self,
        value=None,
        active=None,
        alive=None,
        rendered_frame=None,
    ) -> Observation:
        """
        Create an Observation
        """
        return Observation(
            self.gym_space,
            value=value,
            active=active,
            alive=alive,
            rendered_frame=rendered_frame,
        )

    def serialize(
        self,
        observation: Observation,
    ) -> PbObservation:
        """
        Serialize an Observation to an Observation protobuf message
        """

        serialized_value = None
        if observation.value is not None:
            flat_value = gym.spaces.flatten(self.gym_space, observation.value)
            serialized_value = serialize_ndarray(flat_value)

        serialized_rendered_frame = None
        if observation.rendered_frame is not None:
            serialized_rendered_frame = encode_rendered_frame(
                rendered_frame=observation.rendered_frame, max_size=self.render_width
            )

        return PbObservation(
            value=serialized_value,
            active=observation.active,
            alive=observation.alive,
            rendered_frame=serialized_rendered_frame,
        )

    def deserialize(self, pb_observation: PbObservation) -> Observation:
        """
        Deserialize an Observation from an Observation protobuf message
        """

        return Observation(self.gym_space, pb_observation=pb_observation)

    def create_serialize(
        self,
        value=None,
        active=None,
        alive=None,
        rendered_frame=None,
    ) -> PbObservation:
        """
        Create a serialized Observation
        """
        return self.serialize(
            self.create(
                value=value,
                active=active,
                alive=alive,
                rendered_frame=rendered_frame,
            )
        )
