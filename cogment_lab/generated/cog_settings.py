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

from types import SimpleNamespace

import cogment as _cog

import cogment_lab.generated.data_pb2 as data_pb
import cogment_lab.generated.ndarray_pb2 as ndarray_pb
import cogment_lab.generated.spaces_pb2 as spaces_pb


_player_class = _cog.actor.ActorClass(
            name="player",
            config_type=data_pb.AgentConfig,
            action_space=data_pb.PlayerAction,
            observation_space=data_pb.Observation,
            )


actor_classes = _cog.actor.ActorClassList(_player_class)

trial = SimpleNamespace(config_type=data_pb.TrialConfig)

environment = SimpleNamespace(config_type=data_pb.EnvironmentConfig)
