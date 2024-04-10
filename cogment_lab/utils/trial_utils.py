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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cogment
import gymnasium as gym
import numpy as np
from cogment import ActorParameters
from cogment.datastore import Datastore, DatastoreSample
from tqdm.auto import tqdm

from cogment_lab.generated import cog_settings
from cogment_lab.specs import AgentSpecs
from cogment_lab.utils.grpc import extend_actor_config


def get_actor_params(
    name: str,
    implementation: str,
    agent_specs: AgentSpecs,
    endpoint: str = "grpc://localhost:9002",
    base_params: dict | None = None,
    run_id: str = "run_id",
    seed: int = 0,
) -> ActorParameters:
    """
    Create and return actor parameters for a given actor.

    Args:
        name (str): The name of the actor.
        implementation (str): The implementation type of the actor.
        agent_specs (AgentSpecs): The agent specifications.
        endpoint (str): The endpoint URL, defaults to "grpc://localhost:9002".
        base_params (dict | None): Base parameters for the actor, optional.
        run_id (str): The run ID, defaults to "run_id".
        seed (int): The seed for randomness, defaults to 0.

    Returns:
        ActorParameters: The configured actor parameters.
    """
    if base_params is None:
        base_params = {}
    params = ActorParameters(
        cog_settings,
        name=name,
        class_name="player",
        endpoint=endpoint,
        implementation=implementation,
        config=extend_actor_config(base_params, run_id, agent_specs.serialize(), seed),
    )
    return params


@dataclass
class TrialData:
    """
    Dataclass to store structured trial data for reinforcement learning.

    Attributes:
        observations (np.ndarray | dict[str, np.ndarray] | None): Observations from the trial.
        actions (np.ndarray | dict[str, np.ndarray] | None): Actions taken during the trial.
        rewards (np.ndarray | None): Rewards received during the trial.
        done (np.ndarray | None): Done flags for each step of the trial.
        next_observations (np.ndarray | dict[str, np.ndarray] | None): Next observations after each step.
        last_observation (np.ndarray | dict[str, np.ndarray] | None): The final observation of the trial.
    """

    observations: np.ndarray | dict[str, np.ndarray] | None = field(default=None)
    actions: np.ndarray | dict[str, np.ndarray] | None = field(default=None)
    rewards: np.ndarray | None = field(default=None)
    done: np.ndarray | None = field(default=None)
    next_observations: np.ndarray | dict[str, np.ndarray] | None = field(default=None)
    last_observation: np.ndarray | dict[str, np.ndarray] | None = field(default=None)


def initialize_buffer(space: gym.Space | None, length: int) -> np.ndarray | dict[str, np.ndarray]:
    """
    Initializes a buffer based on the given gym space and length.

    Args:
        space (gym.Space | None): The gym space to base the buffer on. If None, an empty buffer is created.
        length (int): The length of the buffer.

    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: The initialized buffer, either as an ndarray or a dictionary of ndarrays.
    """
    if space is None:
        return np.empty((length,), dtype=np.float32)
    elif isinstance(space, gym.spaces.Dict):
        return {key: initialize_buffer(space[key], length) for key in space.spaces.keys()}  # type: ignore
    elif isinstance(space, gym.spaces.Tuple):
        return {i: initialize_buffer(space[i], length) for i in range(len(space.spaces))}  # type: ignore
    elif isinstance(space, gym.spaces.Text):
        return np.empty((length,), dtype="<U" + str(space.max_length))
    else:  # Simple space
        assert isinstance(space, gym.spaces.Space)
        assert space.shape is not None
        return np.empty((length,) + space.shape, dtype=space.dtype)


def write_to_buffer(
    buffer: np.ndarray | dict[str, np.ndarray],
    data: np.ndarray | dict[str, Any] | bool,
    idx: int,
):
    """
    Writes data to a specified index in the buffer.

    Args:
        buffer (np.ndarray | dict[str, np.ndarray]): The buffer to write data to.
        data (np.ndarray | dict[str, Any]): The data to write.
        idx (int): The index at which to write the data.

    """
    if isinstance(buffer, dict):
        for key in buffer.keys():
            write_to_buffer(buffer[key], data[key], idx)
    else:
        buffer[idx] = data


def extract_data_from_samples(
    samples: list[DatastoreSample],
    agent_specs: AgentSpecs,
    fields: Sequence[str] = (
        "observations",
        "actions",
        "rewards",
        "done",
        "next_observations",
        "last_observation",
    ),
    actor_name: str = "gym",
) -> TrialData:
    """
    Extracts trial data into a TrialData instance from a list of DatastoreSamples.

    Args:
        samples (list[DatastoreSample]): The samples to extract data from.
        fields (Sequence[str]): The fields to extract into the TrialData.
        agent_specs (AgentSpecs): The agent specifications.
        actor_name (str): The name of the actor to extract data for.

    Returns:
       TrialData: The extracted trial data.
    """
    sample_count = len(samples)
    if sample_count == 0:
        raise ValueError("No samples provided")

    cog_observation_space = agent_specs.get_observation_space()
    observation_space = cog_observation_space.gym_space

    cog_action_space = agent_specs.get_action_space()
    action_space = cog_action_space.gym_space

    data = TrialData()
    if "observations" in fields:
        data.observations = initialize_buffer(observation_space, sample_count - 1)  # type: ignore
    if "actions" in fields:
        data.actions = initialize_buffer(action_space, sample_count - 1)  # type: ignore
    if "rewards" in fields:
        data.rewards = initialize_buffer(None, sample_count - 1)  # type: ignore
    if "done" in fields:
        data.done = initialize_buffer(None, sample_count - 1)  # type: ignore
        if sample_count > 1:
            data.done[-1] = True  # type: ignore
    if "next_observations" in fields:
        data.next_observations = initialize_buffer(observation_space, sample_count - 1)
    if "last_observation" in fields:
        data.last_observation = initialize_buffer(observation_space, 1)

    for i, sample in enumerate(samples[:-1]):
        if "observations" in fields:
            obs = cog_observation_space.deserialize(sample.actors_data[actor_name].observation).value
            write_to_buffer(data.observations, obs, i)  # type: ignore
        if "actions" in fields:
            action = cog_action_space.deserialize(sample.actors_data[actor_name].action).value
            write_to_buffer(data.actions, action, i)  # type: ignore
        if "rewards" in fields:
            write_to_buffer(data.rewards, sample.actors_data[actor_name].reward, i)  # type: ignore
        if "done" in fields and i < sample_count - 2:
            write_to_buffer(data.done, False, i)  # type: ignore
        if "next_observations" in fields:
            next_obs = cog_observation_space.deserialize(samples[i + 1].actors_data[actor_name].observation).value
            write_to_buffer(data.next_observations, next_obs, i)  # type: ignore
    if "last_observation" in fields:
        last_obs = (
            agent_specs.get_observation_space().deserialize(samples[-1].actors_data[actor_name].observation).value
        )
        write_to_buffer(data.last_observation, last_obs, 0)  # type: ignore

    return data


def extract_rewards_from_samples(
    samples: list[DatastoreSample],
    actor_name: str = "gym",
) -> np.ndarray:
    """
    Extracts rewards from trial samples into a TrialData instance.

    Args:
        samples (list[DatastoreSample]): The samples to extract rewards from.
        actor_name (str): The name of the actor to extract rewards for.

    Returns:
        TrialData: The extracted rewards.
    """
    sample_count = len(samples)

    rewards = initialize_buffer(None, sample_count - 1)
    assert isinstance(rewards, np.ndarray)

    for i, sample in enumerate(samples[:-1]):
        write_to_buffer(rewards, sample.actors_data[actor_name].reward, i)

    return rewards


def concat_trial_field(
    field_data: list[np.ndarray | dict[str, np.ndarray] | None],
) -> np.ndarray | dict[str, np.ndarray] | None:
    """
    Concatenates a list of fields (either np.ndarray or dict of np.ndarrays) from TrialData instances.
    Filters out None values before concatenation.

    Args:
    field_data: List of fields to be concatenated.

    Returns:
    Concatenated field as np.ndarray or dict of np.ndarrays, or None if all elements are None.
    """
    # Filter out None values
    valid_field_data = [data for data in field_data if data is not None]

    if not valid_field_data:
        return None

    if all(isinstance(data, np.ndarray) for data in valid_field_data):
        return np.concatenate(valid_field_data, axis=0)  # type: ignore
    elif all(isinstance(data, dict) for data in valid_field_data):
        keys = valid_field_data[0].keys()  # type: ignore
        return {key: np.concatenate([data[key] for data in valid_field_data], axis=0) for key in keys}
    else:
        raise TypeError("Inconsistent field types in TrialData list.")


def concatenate(trial_data_list: list[TrialData]) -> TrialData:
    """
    Concatenates a list of TrialData instances into a single TrialData instance.

    Args:
    trial_data_list: List of TrialData instances to be concatenated.

    Returns:
    A single concatenated TrialData instance.
    """
    observations = concat_trial_field([trial.observations for trial in trial_data_list])
    actions = concat_trial_field([trial.actions for trial in trial_data_list])
    rewards = concat_trial_field([trial.rewards for trial in trial_data_list])
    done = concat_trial_field([trial.done for trial in trial_data_list])
    next_observations = concat_trial_field([trial.next_observations for trial in trial_data_list])

    # Handle 'last_observation' separately
    last_trial = trial_data_list[-1]
    last_observation = (
        last_trial.last_observation if last_trial.last_observation is not None else last_trial.next_observations[-1]  # type: ignore
    )

    return TrialData(
        observations=observations,
        actions=actions,
        rewards=rewards,  # type: ignore
        done=done,  # type: ignore
        next_observations=next_observations,
        last_observation=last_observation,
    )


async def format_data_multiagent(
    datastore: Datastore,
    trial_id: str,
    actor_agent_specs: dict[str, AgentSpecs],
    fields: Sequence[str] = (
        "observations",
        "actions",
        "rewards",
        "done",
        "next_observations",
        "last_observation",
    ),
    use_tqdm: bool = False,
    tqdm_kwargs: dict[str, Any] | None = None,
) -> dict[str, TrialData]:
    """
    Formats trial data from a multiagent Cogment trial into structured formats for reinforcement learning.

    Args:
        datastore (Datastore): The datastore to fetch trial data from.
        trial_id (str): The identifier of the trial.
        actor_agent_specs (dict[str, EnvironmentSpecs]): A dictionary mapping actor IDs to their environment specifications.
        fields (List[str]): The list of fields to include in the formatted data.
        tqdm_kwargs (dict[str, Any] | None): Optional keyword arguments to pass to tqdm.

    Returns:
        dict[str, TrialData]: A dictionary mapping actor IDs to their formatted trial data.
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    trials = []
    while len(trials) == 0:
        try:
            trials = await datastore.get_trials([trial_id])
        except cogment.CogmentError:
            continue

    # Initialize a dictionary to store samples for each actor
    actor_samples = {actor_id: [] for actor_id in actor_agent_specs.keys()}
    actor_reward_samples = {actor_id: [] for actor_id in actor_agent_specs.keys()}

    # Get all samples
    async for sample in tqdm(datastore.all_samples(trials), disable=not use_tqdm, **tqdm_kwargs):  # type: ignore
        for actor_id in actor_agent_specs.keys():
            # Add the sample to the list for an actor if the observation for that actor is not None
            if (
                sample.actors_data.get(actor_id)
                and sample.actors_data[actor_id].observation is not None
                and sample.actors_data[actor_id].observation.value is not None
                and sample.actors_data[actor_id].observation.value.raw_data != b""
            ):
                actor_samples[actor_id].append(sample)

            if (
                sample.actors_data.get(actor_id)
                and sample.actors_data[actor_id].reward is not None
                and sample.actors_data[actor_id].reward == sample.actors_data[actor_id].reward  # Check for NaN
            ):
                actor_reward_samples[actor_id].append(sample)

    # Extract data for each actor
    actor_data = {}

    for actor_id, samples in actor_samples.items():
        actor_data[actor_id] = extract_data_from_samples(
            samples, actor_agent_specs[actor_id], fields, actor_name=actor_id
        )

    for actor_id, reward_samples in actor_reward_samples.items():
        actor_data[actor_id].rewards = extract_rewards_from_samples(reward_samples, actor_name=actor_id)

    return actor_data
