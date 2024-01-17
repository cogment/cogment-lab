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

import asyncio
import datetime

from cogment_lab.actors import ConstantActor
from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment


async def main():
    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    cenv = GymEnvironment(
        env_id="MountainCar-v0",
        render=True,
        make_kwargs={"max_episode_steps": 10},
    )

    await cog.run_env(env=cenv, env_name="mcar", port=9011, log_file="env.log")

    # Create a model using coltra

    constant_actor = ConstantActor(1)

    await cog.run_actor(actor=constant_actor, actor_name="constant", port=9022, log_file="actor-constant.log")

    # Estimate random agent performance

    trial_id = await cog.start_trial(
        env_name="mcar",
        session_config={"render": False},
        actor_impls={
            "gym": "constant",
        },
    )
    multi_data = await cog.get_trial_data(trial_id=trial_id)
    data = multi_data["gym"]

    # mean_reward = np.mean([sum(e.rewards) for e in episodes])
    print(f"Reward shape: {data.rewards.shape}")
    print(f"Rewards: {data.rewards}")
    print(f"Observations: {data.observations}")
    print(f"Last observation: {data.last_observation}")
    print(f"Actions: {data.actions}")


if __name__ == "__main__":
    asyncio.run(main())
