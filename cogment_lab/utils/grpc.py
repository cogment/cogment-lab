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

from __future__ import annotations

from google.protobuf.json_format import ParseDict

from cogment_lab.generated import data_pb2


def extend_actor_config(
    actor_config_template: dict,
    run_id: str,
    agent_specs: data_pb2.AgentSpecs,  # type: ignore
    seed: int,
) -> data_pb2.AgentConfig:  # type: ignore
    """Extends an actor configuration template with additional parameters.

    Args:
        actor_config_template: Template for actor configuration, possibly None.
        run_id: Identifier for the run.
        agent_specs: Specifications for the environment.
        seed: Seed for random number generation.

    Returns:
        An instance of AgentConfig with extended configuration.
    """
    config = data_pb2.AgentConfig()  # type: ignore
    if actor_config_template:
        ParseDict(actor_config_template, config)
    config.run_id = run_id
    config.agent_specs.CopyFrom(agent_specs)
    config.seed = seed
    return config


def create_value(val: str | int | float) -> data_pb2.Value:  # type: ignore
    """Creates a Value protobuf message from a Python value.

    Args:
        val: The input string, integer, or float.

    Returns:
        A Value protobuf message containing the input value.
    """
    value_message = data_pb2.Value()  # type: ignore
    if isinstance(val, str):
        value_message.string_value = val
    elif isinstance(val, int):
        value_message.int_value = val
    elif isinstance(val, float):
        value_message.float_value = val
    else:
        raise ValueError("Unsupported type")
    return value_message


def get_env_config(
    run_id: str | None = None,
    render: bool | None = None,
    render_width: int | None = None,
    seed: int | None = None,
    flatten: bool | None = None,
    reset_args_dict: dict[str, str | int | float] | None = None,
) -> data_pb2.EnvironmentConfig:  # type: ignore
    """Generates an EnvironmentConfig protobuf message.

    Args:
        run_id: Identifier for the run.
        render: Whether to render the environment.
        render_width: Render width if rendering.
        seed: Random seed.
        flatten: Whether to flatten the observation.
        reset_args_dict: Dictionary of reset argument values.

    Returns:
        An EnvironmentConfig protobuf message.
    """
    if reset_args_dict is None:
        reset_args_dict = {}
    env_config = data_pb2.EnvironmentConfig()  # type: ignore

    env_config.run_id = run_id
    env_config.render = render
    env_config.render_width = render_width
    env_config.seed = seed
    env_config.flatten = flatten

    for key, val in reset_args_dict.items():
        env_config.reset_args[key].CopyFrom(create_value(val))

    return env_config
