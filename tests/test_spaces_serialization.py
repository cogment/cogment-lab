# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from cogment_lab.specs.ndarray_serialization import SerializationFormat
from cogment_lab.specs.spaces_serialization import serialize_gym_space


# pylint: disable=no-member


def test_serialize_custom_observation_space():
    """Test serialization of gym spaces of type:
    Dict, Discrete, Box, MultiDiscrete, MultiBinary.
    """
    gym_space = Dict(
        {
            "ext_controller": MultiDiscrete([5, 2, 2]),
            "inner_state": Dict(
                {
                    "charge": Discrete(100),
                    "system_checks": MultiBinary(10),
                    "system_checks_seq": MultiBinary([2, 5, 10]),
                    "system_checks_array": MultiBinary(np.array([2, 5, 10])),
                    "job_status": Dict(
                        {
                            "task": Discrete(5),
                            "progress": Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
            "tuple_state": Tuple([Discrete(i) for i in range(1, 4)]),
        }
    )

    pb_space = serialize_gym_space(gym_space, serialization_format=SerializationFormat.STRUCTURED)

    assert len(pb_space.dict.spaces) == 3
    assert pb_space.dict.spaces[0].key == "ext_controller"
    assert pb_space.dict.spaces[0].space.multi_discrete.nvec.shape == [3]

    assert pb_space.dict.spaces[1].key == "inner_state"
    assert len(pb_space.dict.spaces[1].space.dict.spaces) == 5

    assert pb_space.dict.spaces[1].space.dict.spaces[0].key == "charge"
    assert pb_space.dict.spaces[1].space.dict.spaces[0].space.discrete.n == 100

    assert pb_space.dict.spaces[1].space.dict.spaces[1].key == "job_status"
    assert len(pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces) == 2

    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].key == "progress"
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].space.box.low.double_data[
        0
    ] == pytest.approx(0.0)
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].space.box.high.double_data[
        0
    ] == pytest.approx(100.0)

    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[1].key == "task"
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[1].space.discrete.n == 5

    assert pb_space.dict.spaces[1].space.dict.spaces[2].key == "system_checks"
    assert pb_space.dict.spaces[1].space.dict.spaces[2].space.multi_binary.n.shape == [1]

    assert pb_space.dict.spaces[1].space.dict.spaces[3].key == "system_checks_array"
    assert pb_space.dict.spaces[1].space.dict.spaces[3].space.multi_binary.n.shape == [3]

    assert pb_space.dict.spaces[1].space.dict.spaces[4].key == "system_checks_seq"
    assert pb_space.dict.spaces[1].space.dict.spaces[4].space.multi_binary.n.shape == [3]

    assert pb_space.dict.spaces[2].key == "tuple_state"
    assert pb_space.dict.spaces[2].space.tuple.spaces[0].space.discrete.n == 1
    assert pb_space.dict.spaces[2].space.tuple.spaces[1].space.discrete.n == 2
    assert pb_space.dict.spaces[2].space.tuple.spaces[2].space.discrete.n == 3
