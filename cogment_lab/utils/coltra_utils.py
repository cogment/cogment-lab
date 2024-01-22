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

import numpy as np
import torch
from coltra import Agent
from coltra.buffers import Action, Observation, OnPolicyRecord

from cogment_lab.utils.trial_utils import TrialData


def convert_trial_data_to_coltra(trial_data: TrialData, agent: Agent) -> OnPolicyRecord:
    """Convert TrialData to OnPolicyRecord.

    Args:
        trial_data (TrialData): TrialData instance
        model (Agent): Model instance to evaluate values

    Returns:
        OnPolicyRecord: Converted OnPolicyRecord instance
    """
    obs = trial_data.observations
    action = trial_data.actions
    reward = trial_data.rewards
    done = trial_data.done
    # state = None  # Assuming 'state' is not provided in TrialData

    # last_value = agent.act(Observation(vector=trial_data.last_observation), get_value=True)[2]["value"]
    # value = agent.act(Observation(vector=trial_data.observations), get_value=True)[2]["value"]

    last_value, _ = agent.value(Observation(vector=trial_data.last_observation), ())  # type: ignore
    value, _ = agent.value(Observation(vector=trial_data.observations), ())  # type: ignore

    last_value = last_value.detach().squeeze(-1).cpu().numpy()
    value = value.detach().squeeze(-1).cpu().numpy()

    # Check if required fields are not None
    if obs is None or action is None or reward is None or done is None:
        raise ValueError("Missing required fields in TrialData for conversion")

    # Create an OnPolicyRecord instance with the mapped fields
    on_policy_record = OnPolicyRecord(
        obs=Observation(vector=obs).tensor(),  # type: ignore
        action=Action(discrete=action).tensor(),  # type: ignore
        reward=torch.tensor(reward.astype(np.float32)),
        value=torch.tensor(value.astype(np.float32)),
        done=torch.tensor(done.astype(np.float32)),
        last_value=torch.tensor(last_value.astype(np.float32)),
    )

    return on_policy_record
