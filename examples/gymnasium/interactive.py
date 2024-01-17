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

from cogment_lab.actors import RandomActor, ConstantActor
from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.utils.runners import process_cleanup
from cogment_lab.utils.trial_utils import format_data_multiagent


async def main():
    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    # Launch an environment in a subprocess

    cenv = GymEnvironment(env_id="CartPole-v1", render=True)

    print("Starting env")

    await cog.run_env(env=cenv, env_name="cartpole", port=9001, log_file="env.log")

    # Launch two dummy actors in subprocesses

    print("Starting actors")

    random_actor = RandomActor(cenv.env.action_space)
    constant_actor = ConstantActor(0)

    await cog.run_actor(actor=random_actor, actor_name="random", port=9021, log_file="actor-random.log")

    await cog.run_actor(actor=constant_actor, actor_name="constant", port=9022, log_file="actor-constant.log")

    # Start a trial

    print("Starting trial")

    trial_id = await cog.start_trial(
        env_name="cartpole",
        session_config={"render": True},
        actor_impls={
            "gym": "random",
        },
    )

    print("Waiting for trial to finish")

    data = await format_data_multiagent(datastore=cog.datastore, trial_id=trial_id, actor_agent_specs=cenv.agent_specs)

    print(data)
    return


if __name__ == "__main__":
    asyncio.run(main())
