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

import matplotlib.pyplot as plt
from coltra import HomogeneousGroup
from coltra.models import MLPModel
from coltra.policy_optimization import CrowdPPOptimizer
from tqdm import trange

from cogment_lab.actors import ColtraActor
from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.utils.coltra_utils import convert_trial_data_to_coltra
from cogment_lab.utils.runners import process_cleanup
from cogment_lab.utils.trial_utils import concatenate


async def main():
    process_cleanup()

    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    # We'll train on CartPole-v1

    cenv = GymEnvironment(
        env_id="CartPole-v1",
        render=False,
    )

    await cog.run_env(env=cenv, env_name="cartpole", port=9001, log_file="env.log")

    print("Env started")

    # Create a model using coltra

    model = MLPModel(
        config={
            "hidden_sizes": [64, 64],
        },
        observation_space=cenv.env.observation_space,
        action_space=cenv.env.action_space,
    )

    actor = ColtraActor(model=model)

    cog.run_local_actor(actor=actor, actor_name="coltra", port=9021, log_file="actor.log")

    print("Actor started")

    ppo = CrowdPPOptimizer(
        HomogeneousGroup(actor.agent),
        config={
            "gae_lambda": 0.95,
            "minibatch_size": 128,
        },
    )

    all_rewards = []

    for t in (pbar := trange(100)):
        num_steps = 0
        episodes = []
        while num_steps < 1000:  # Collect at least 1000 steps per training iteration
            trial_id = await cog.start_trial(
                env_name="cartpole",
                session_config={"render": False},
                actor_impls={
                    "gym": "coltra",
                },
            )
            multi_data = await cog.get_trial_data(trial_id=trial_id)
            data = multi_data["gym"]
            episodes.append(data)
            num_steps += len(data.rewards)

        all_data = concatenate(episodes)

        # Preprocess data

        record = convert_trial_data_to_coltra(all_data, actor.agent)

        # Run a PPO step
        metrics = ppo.train_on_data({"crowd": record}, shape=(1, len(record.reward)))

        mean_reward = metrics["crowd/mean_episode_reward"]
        all_rewards.append(mean_reward)
        pbar.set_description(f"mean_reward: {mean_reward:.3}")

    plt.plot(all_rewards)
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
