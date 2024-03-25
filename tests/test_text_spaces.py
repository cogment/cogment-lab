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
import pytest

import tests.shared
from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.specs.ndarray_serialization import (
    SerializationFormat,
    deserialize_ndarray,
    serialize_ndarray,
)


@pytest.mark.parametrize(
    "data",
    [
        "foo",
        "bar",
        "hello world",
        ["foo", "bar"],
        ["hello", "world"],
        ["one", "two", "three"],
        [["11", "12", "13"], ["21", "22", "23"], ["31", "32", "33"]],
    ],
)
def test_serialize_ndarray(data):
    arr = np.array(data)
    serialized = serialize_ndarray(arr, SerializationFormat.STRUCTURED)
    deserialized = deserialize_ndarray(serialized)
    assert np.array_equal(arr, deserialized)


def test_text_env():
    env = gym.make("Echo-v0")
    obs, info = env.reset(seed=0)
    assert isinstance(obs, str)

    obs, reward, terminated, truncated, info = env.step("not the same")

    assert isinstance(obs, str)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False

    obs, reward, terminated, truncated, info = env.step(obs)

    assert isinstance(obs, str)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False


@pytest.mark.asyncio
async def test_echo():
    """Test the echo environment."""

    cog = Cogment(log_dir="logs")

    cenv = GymEnvironment(env_id="tests.shared:Echo-v0", render=False)

    await cog.run_env(env=cenv, env_name="echo_env", port=9011, log_file="echo_env.log")

    actor = tests.shared.NoisyEchoActor(noise_prob=0.8)

    await cog.run_actor(
        actor=actor,
        actor_name="echo_actor",
        port=9021,
        log_file="actor-echo.log",
    )

    trial_id = await cog.start_trial(
        env_name="echo_env",
        session_config={"render": False},
        actor_impls={
            "gym": "echo_actor",
        },
    )

    data = await cog.get_trial_data(trial_id=trial_id)

    assert isinstance(data, dict)
    assert isinstance(data["gym"].observations, np.ndarray)
    assert isinstance(data["gym"].actions, np.ndarray)
    assert isinstance(data["gym"].rewards, np.ndarray)

    await cog.cleanup()
