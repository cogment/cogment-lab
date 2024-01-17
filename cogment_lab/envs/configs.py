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

"""
Module for environment configuration.

Defines data structures for specifying environment configurations.
"""

from __future__ import annotations

from typing import Any, TypedDict


class EnvConfig(TypedDict, total=False):
    """
    Configuration for a single environment.

    Attributes:
        env_type: Type of the environment.
        env_name: Name of the environment.
        cogment_env: Cogment environment ID to use, instead of env_id/registration/make_args.
        env_id: Environment ID.
        registration: Environment registration details.
        make_args: Arguments for make() to construct the environment.
        reset_options: Reset options for the environment.
        render: Whether to render the environment.
    """

    env_type: str
    env_name: str

    # Either this...
    cogment_env: str | None

    # ...or all of this
    env_id: str | None
    registration: str | None
    make_args: dict[str, Any]
    reset_options: dict[str, Any]
    render: bool


class EnvRunnerConfig(TypedDict):
    """
    Configuration for an environment runner.

    Attributes:
        envs: List of EnvConfig, one for each environment.
        port: Port number for the runner.
    """

    envs: list[EnvConfig]
    port: int
