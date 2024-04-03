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
