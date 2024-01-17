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

from cogment_lab import Cogment
from cogment_lab.actors import ConstantActor, RandomActor
from cogment_lab.envs import GymEnvironment
from cogment_lab.utils.runners import process_cleanup


LUNAR_LANDER_ACTIONS = ["no-op", "ArrowRight", "ArrowUp", "ArrowLeft"]


async def main():
    # Create the global process manager
    process_cleanup()

    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    # Launch the environment
    env = GymEnvironment(
        env_id="LunarLander-v2",  # ID of a Gymnasium environment
        render=True,  # True if we want to see the rendering at some point
    )

    await cog.run_env(
        env=env,
        env_name="lunar",
        port=9011,  # Typically, we use ports 901x for environments and 902x for actors
        log_file="env.log",
    )

    # Launch a constant actor
    constant_actor = ConstantActor(0)

    await cog.run_actor(actor=constant_actor, actor_name="constant", port=9021, log_file="random.log")

    # Launch a random actor
    random_actor = RandomActor(env.env.action_space)

    await cog.run_actor(actor=random_actor, actor_name="random", port=9022, log_file="constant.log")

    # Launch an episode
    episode_id = await cog.start_trial(
        env_name="lunar",  # Which environment
        actor_impls={"gym": "random"},  # Which actor(s) will act
    )

    # Compute the total reward of the episode
    data = await cog.get_trial_data(trial_id=episode_id)
    random_reward = data["gym"].rewards.sum()

    print(f"Random agent's reward: {random_reward}")

    # Launch a human actor UI
    await cog.run_web_ui(actions=LUNAR_LANDER_ACTIONS, log_file="human.log", fps=60)

    episode_id = await cog.start_trial(env_name="lunar", actor_impls={"gym": "web_ui"}, session_config={"render": True})

    print("Go to http://localhost:8000 in your browser and see how well you do!")

    data = await cog.get_trial_data(trial_id=episode_id)

    human_reward = data["gym"].rewards.sum()

    if human_reward > random_reward:
        print(f"Good job! You beat a random agent with a score of {human_reward}!")
    else:
        print(f"Awkward... You lost with a score of {human_reward}...")


if __name__ == "__main__":
    asyncio.run(main())
