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

import numpy as np
import pytest

from cogment_lab.actors import ConstantActor
from cogment_lab.envs.pettingzoo import AECEnvironment
from cogment_lab.process_manager import Cogment


@pytest.mark.asyncio
async def test_pong():
    """Test the cartpole environment."""

    cog = Cogment(log_dir="logs")

    cenv = AECEnvironment(env_path="pettingzoo.butterfly.cooperative_pong_v5.env", render=False)

    await cog.run_env(env=cenv, env_name="pong", port=9012, log_file="env.log")

    constant_actor = ConstantActor(1)

    await cog.run_actor(
        actor=constant_actor,
        actor_name="constant",
        port=9022,
        log_file="actor-constant.log",
    )

    trial_id = await cog.start_trial(
        env_name="pong",
        session_config={"render": False},
        actor_impls={
            "paddle_0": "constant",
            "paddle_1": "constant",
        },
    )

    data = await cog.get_trial_data(trial_id=trial_id, env_name="pong")

    for agent_name in ["paddle_0", "paddle_1"]:
        assert isinstance(data, dict)
        assert isinstance(data[agent_name].observations, np.ndarray)
        assert isinstance(data[agent_name].actions, np.ndarray)
        assert isinstance(data[agent_name].rewards, np.ndarray)

    await cog.cleanup()
