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

import datetime
import asyncio

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from coltra import HomogeneousGroup
from coltra.buffers import Observation
from coltra.models import MLPModel
from coltra.policy_optimization import CrowdPPOptimizer
from tqdm import trange

from cogment_lab.actors import ColtraActor, ConstantActor
from cogment_lab.envs import AECEnvironment
from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.utils.coltra_utils import convert_trial_data_to_coltra
from cogment_lab.utils.runners import process_cleanup
from cogment_lab.utils.trial_utils import concatenate


async def main():
    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    cenv = AECEnvironment(
        env_path="pettingzoo.butterfly.cooperative_pong_v5.env", render=False, make_kwargs={"max_cycles": 20}
    )

    await cog.run_env(env=cenv, env_name="pong", port=9011, log_file="env.log")

    # Create a model using coltra

    constant_actor = ConstantActor(1)

    await cog.run_actor(actor=constant_actor, actor_name="constant", port=9022, log_file="actor-constant.log")

    # Estimate random agent performance

    trial_id = await cog.start_trial(
        env_name="pong",
        session_config={"render": True},
        actor_impls={
            "paddle_0": "constant",
            "paddle_1": "constant",
        },
    )

    data = await cog.get_trial_data(trial_id=trial_id)

    # mean_reward = np.mean([sum(e.rewards) for e in episodes])
    print(f"Reward shape: {data['paddle_0'].rewards.shape}")
    print(f"Rewards: {data['paddle_0'].rewards}")
    print(f"Action shape: {data['paddle_0'].actions.shape}")
    print(f"Actions: {data['paddle_0'].actions}")

    # Other agent
    print(f"Reward shape: {data['paddle_1'].rewards.shape}")
    print(f"Rewards: {data['paddle_1'].rewards}")
    print(f"Action shape: {data['paddle_1'].actions.shape}")
    print(f"Actions: {data['paddle_1'].actions}")


if __name__ == "__main__":
    asyncio.run(main())
