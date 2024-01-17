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
import logging

import cogment
import yaml

from cogment_lab.core import BaseEnv, NativeEnv
from cogment_lab.envs.configs import EnvConfig
from cogment_lab.envs.environment import GymEnvironment
from cogment_lab.generated import cog_settings
from cogment_lab.utils import import_object


# log = logging.getLogger(__name__)


def get_environment(config: EnvConfig) -> BaseEnv:
    """Given a config, generates an impl for a cogment environment"""
    env_type = config.get("env_type")
    if env_type is None:
        raise ValueError("env_type is not provided in config")

    if env_type == "gymnasium":
        env = GymEnvironment(config)
    elif env_type == "cogment":
        impl_path = config["cogment_env"]
        impl = import_object(impl_path)
        env = NativeEnv(impl=impl)
    else:
        raise NotImplementedError(f"Invalid env_type: {env_type}")

    return env


async def create_envs(env_configs: list[EnvConfig], port: int):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")

    for config in env_configs:
        env = get_environment(config)
        context.register_environment(impl=env.impl, impl_name=config["env_name"])

    await context.serve_all_registered(cogment.ServedEndpoint(port=port))


def env_main(config_path: str):
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    logging.info(config)
    asyncio.run(create_envs(env_configs=config["environments"], port=config["port"]))
