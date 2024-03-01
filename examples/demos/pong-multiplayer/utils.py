import numpy as np
import torch
from coltra import CAgent, DAgent, Observation
from coltra.envs.spaces import ObservationSpace
from coltra.models import BaseModel
from coltra.models.mlp_models import FlattenMLPModel
from gymnasium import Space
from gymnasium.spaces import Box
from pettingzoo.butterfly.cooperative_pong_v5 import parallel_env
from supersuit import dtype_v0, normalize_obs_v0

from cogment_lab.core import CogmentActor


class FloatImageMLPModel(FlattenMLPModel):
    def __init__(self, config: dict, observation_space: ObservationSpace, action_space: Space):
        assert "image" in observation_space.spaces, "ImageMLPModel requires an observation space with image"

        vector_size = observation_space.vector.shape[0] if "vector" in observation_space.spaces else 0
        image_size = np.prod(observation_space.spaces["image"].shape)
        new_vector_size = vector_size + image_size

        new_observation_space = ObservationSpace({"vector": Box(-np.inf, np.inf, (new_vector_size,))})

        super().__init__(config, new_observation_space, action_space)

    def _flatten(self, obs: Observation) -> Observation:
        if not hasattr(obs, "image"):
            return obs
        image: torch.Tensor = obs.image

        if len(image.shape) == 3:  # no batch
            dim = 0
        else:  # image.shape == 4, batch
            dim = 1

        vector = torch.flatten(image, start_dim=dim)

        if hasattr(obs, "vector"):
            vector = torch.cat([obs.vector, vector], dim=dim)

        return Observation(vector=vector.to(torch.float32))


class ColtraImageActor(CogmentActor):
    def __init__(self, model: BaseModel):
        super().__init__(model)
        self.model = model
        self.agent = DAgent(self.model) if self.model.discrete else CAgent(self.model)

    async def act(self, observation: np.ndarray, rendered_frame=None):
        obs = Observation(image=(observation / 255.0).astype(np.float32))
        action, _, _ = self.agent.act(obs)
        return action.discrete


def PongEnv(*args, **kwargs):
    env = parallel_env(*args, **kwargs)
    env = dtype_v0(env, np.float32)
    env = normalize_obs_v0(env, env_max=255.0)
    return env
