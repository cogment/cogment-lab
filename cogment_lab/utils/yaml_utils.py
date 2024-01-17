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

import gymnasium as gym
import numpy as np
import yaml


def gym_space_constructors():
    """Registers YAML constructors for Gym spaces.

    This allows Gym spaces to be created automatically from YAML files
    by registering constructors for each space type.
    """

    def box_constructor(loader, node):
        """YAML constructor for Box spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Box space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        return gym.spaces.Box(
            low=np.array(values.get("low", -np.inf)),
            high=np.array(values.get("high", np.inf)),
            shape=values.get("shape", None),
            dtype=values.get("dtype", np.float32),
            seed=values.get("seed", None),
        )

    def discrete_constructor(loader, node):
        """YAML constructor for Discrete spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Discrete space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        return gym.spaces.Discrete(n=values["n"], seed=values.get("seed", None), start=values.get("start", 0))

    def multibinary_constructor(loader, node):
        """YAML constructor for MultiBinary spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A MultiBinary space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        return gym.spaces.MultiBinary(n=values["n"], seed=values.get("seed", None))

    def multidiscrete_constructor(loader, node):
        """YAML constructor for MultiDiscrete spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A MultiDiscrete space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        return gym.spaces.MultiDiscrete(
            nvec=np.array(values["nvec"]),
            dtype=values.get("dtype", np.int64),
            seed=values.get("seed", None),
            start=values.get("start", None),
        )

    def text_constructor(loader, node):
        """YAML constructor for Text spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Text space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        return gym.spaces.Text(
            max_length=values["max_length"],
            min_length=values.get("min_length", 1),
            charset=values.get("charset", "alphanumeric"),
            seed=values.get("seed", None),
        )

    def dict_constructor(loader, node):
        """YAML constructor for Dict spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Dict space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        spaces = values.pop("spaces", None)
        seed = values.pop("seed", None)
        return gym.spaces.Dict(spaces=spaces, seed=seed, **values)

    def tuple_constructor(loader, node):
        """YAML constructor for Tuple spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Tuple space constructed from the YAML node.
        """
        values = loader.construct_sequence(node)
        spaces = values.pop("spaces", None)
        seed = values.pop("seed", None)
        return gym.spaces.Tuple(spaces=spaces, seed=seed)

    def sequence_constructor(loader, node):
        """YAML constructor for Sequence spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Sequence space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        space = values.get("space")
        seed = values.get("seed", None)
        stack = values.get("stack", False)
        return gym.spaces.Sequence(space=space, seed=seed, stack=stack)

    def graph_constructor(loader, node):
        """YAML constructor for Graph spaces.

        Args:
            loader: The YAML loader.
            node: The YAML node.

        Returns:
            A Graph space constructed from the YAML node.
        """
        values = loader.construct_mapping(node)
        node_space = values.pop("node_space")
        edge_space = values.pop("edge_space", None)
        seed = values.pop("seed", None)
        return gym.spaces.Graph(node_space=node_space, edge_space=edge_space, seed=seed)

    # Register constructors
    yaml.add_constructor("!Box", box_constructor)
    yaml.add_constructor("!Discrete", discrete_constructor)
    yaml.add_constructor("!MultiBinary", multibinary_constructor)
    yaml.add_constructor("!MultiDiscrete", multidiscrete_constructor)
    yaml.add_constructor("!Text", text_constructor)

    yaml.add_constructor("!Dict", dict_constructor)
    yaml.add_constructor("!Tuple", tuple_constructor)
    yaml.add_constructor("!Graph", graph_constructor)
    yaml.add_constructor("!Sequence", sequence_constructor)
    yaml.add_constructor("!Tuple", tuple_constructor)
