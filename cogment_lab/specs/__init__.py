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

from cogment_lab.specs.action_space import Action, ActionSpace
from cogment_lab.specs.encode_rendered_frame import (
    decode_rendered_frame,
    encode_rendered_frame,
)
from cogment_lab.specs.environment_specs import AgentSpecs
from cogment_lab.specs.ndarray_serialization import (
    SerializationFormat,
    deserialize_ndarray,
    serialize_ndarray,
)
from cogment_lab.specs.observation_space import Observation, ObservationSpace
from cogment_lab.specs.spaces_serialization import (
    deserialize_space,
    serialize_gym_space,
)
