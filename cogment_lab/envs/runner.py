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

"""Registers an environment with Cogment and runs the Cogment server"""

from __future__ import annotations

import asyncio
import logging
from multiprocessing import Queue

import cogment

from cogment_lab.core import BaseEnv
from cogment_lab.generated import cog_settings
from cogment_lab.utils.runners import setup_logging


async def register_env(env: BaseEnv, env_name: str, signal_queue: Queue, port: int = 9001):
    """Registers an environment with Cogment and runs the Cogment server

    Args:
        env (BaseEnv): The environment to register
        env_name (str): The name to register the environment under
        signal_queue (Queue): A queue to signal when server has started
        port (int, optional): The port for the Cogment server. Defaults to 9001.

    """
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")

    context.register_environment(impl=env.impl, impl_name=env_name)
    logging.info(f"Registered environment {env_name} with cogment")

    serve = context.serve_all_registered(cogment.ServedEndpoint(port=port))
    signal_queue.put(True)
    await serve


def env_runner(
    env_class: type,
    env_args: tuple,
    env_kwargs: dict,
    env_name: str,
    signal_queue: Queue,
    port: int = 9001,
    log_file: str | None = None,
):
    """Given an environment, runs it

    Args:
        env_class (type): The environment class to instantiate
        env_args (tuple): Positional arguments for the environment
        env_kwargs (dict): Keyword arguments for the environment
        env_name (str): The name to register the environment under
        signal_queue (Queue): A queue to signal when server has started
        port (int, optional): The port for the Cogment server. Defaults to 9001.
        log_file (str | None, optional): File path to write logs to. Defaults to None.
    """
    if log_file:
        setup_logging(log_file)
    env = env_class(*env_args, **env_kwargs)

    asyncio.run(register_env(env, env_name, signal_queue, port))
