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
import random

import numpy as np
import torch
import wandb
from coltra.models import FCNetwork
from torch import optim
from tqdm import trange
from typarse import BaseParser

from cogment_lab.actors.nn_actor import NNActor
from cogment_lab.envs import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.utils.runners import process_cleanup
from shared import ReplayBuffer, dqn_loss, get_current_eps


class Parser(BaseParser):
    wandb_project: str = "test"
    wandb_name: str = "test"

    env_name: str = "LunarLander-v2"

    batch_size: int = 128
    gamma: float = 0.99
    replay_buffer_capacity: int = 50000
    learning_rate: float = 6.3e-4
    num_episodes: int = 500
    seed: int = 0


async def main():
    args = Parser()

    process_cleanup()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_name)

    wandb.config.batch_size = args.batch_size
    wandb.config.gamma = args.gamma
    wandb.config.replay_buffer_capacity = args.replay_buffer_capacity
    wandb.config.learning_rate = args.learning_rate
    wandb.config.num_episodes = args.num_episodes
    wandb.config.seed = args.seed
    wandb.config.env_name = args.env_name

    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    cenv = GymEnvironment(env_id=args.env_name, reinitialize=True, render=True)

    obs_len = cenv.env.observation_space.shape[0]

    await cog.run_env(cenv, "lunar", port=9021, log_file="env-gym.log")

    # Create and run the learner network

    replay_buffer = ReplayBuffer(args.replay_buffer_capacity, obs_len)

    # Run the agent
    network = FCNetwork(
        input_size=obs_len,
        output_sizes=[cenv.env.action_space.n],
        hidden_sizes=[256, 256],
        activation="tanh",
    )

    actor = NNActor(network, "cpu")
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

    cog.run_local_actor(actor, "dqn", port=9012, log_file="dqn.log")

    total_timesteps = 0
    ep_rewards = []

    for episode in (pbar := trange(args.num_episodes)):
        actor.set_eps(get_current_eps(episode))

        trial_id = await cog.start_trial(
            env_name="lunar",
            actor_impls={"gym": "dqn"},
            session_config={"render": True, "seed": episode},
        )

        trial_data_task = asyncio.create_task(cog.get_trial_data(trial_id))

        gradient_updates = 0

        trial_data = await trial_data_task

        # Logging
        dqn_data = trial_data["gym"]

        total_reward = dqn_data.rewards.sum()
        pbar.set_description(f"mean_reward: {total_reward:.3}")
        ep_rewards.append(total_reward)

        total_timesteps += len(dqn_data.rewards)

        # Add data to replay buffer

        for t in range(len(dqn_data.done)):
            state = dqn_data.observations[t]
            action = dqn_data.actions[t]

            reward = dqn_data.rewards[t]
            next_state = dqn_data.next_observations[t]
            done = dqn_data.done[t]

            replay_buffer.push(state, action, reward, next_state, done)

            # Train, once per datapoint

            if len(replay_buffer) > args.batch_size:
                batch = replay_buffer.sample(args.batch_size)

                loss = dqn_loss(network, batch, args.gamma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                gradient_updates += 1

        log_dict = {
            "episode": episode,
            "reward": total_reward,
            "ep_length": len(dqn_data.rewards),
            "total_timesteps": total_timesteps,
            "gradient_updates": gradient_updates,
        }

        wandb.log(log_dict)


if __name__ == "__main__":
    asyncio.run(main())
