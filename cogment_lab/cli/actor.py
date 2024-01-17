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

from cogment_lab.actors.configs import AgentConfig
from cogment_lab.core import BaseActor, NativeActor
from cogment_lab.generated import cog_settings
from cogment_lab.utils import import_object
from cogment_lab.utils.yaml_utils import gym_space_constructors


log = logging.getLogger(__name__)


def get_actor(agent_config: AgentConfig) -> BaseActor:
    agent_type = agent_config.get("agent_type")
    if agent_type is None:
        raise ValueError("agent_type is not provided in config")

    if agent_type == "cogment":
        impl_path = agent_config["cogment_actor"]
        impl = import_object(impl_path)
        agent = NativeActor(impl=impl)
    elif agent_type == "custom":
        agent_class = agent_config["agent_class"]
        agent_kwargs = agent_config["agent_kwargs"]
        cls = import_object(agent_class)
        agent = cls(**agent_kwargs)
    else:
        raise NotImplementedError(f"Invalid agent_type: {agent_type}")

    return agent


async def create_agents(agent_configs: list[AgentConfig], port: int):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")

    for config in agent_configs:
        agent = get_actor(config)
        context.register_actor(impl=agent.impl, impl_name=config["agent_name"])

    await context.serve_all_registered(cogment.ServedEndpoint(port=port))


def actor_main(config_path: str):
    gym_space_constructors()
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    log.info(config)
    asyncio.run(create_agents(agent_configs=config["agents"], port=config["port"]))
