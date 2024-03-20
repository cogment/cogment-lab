import asyncio
import json
import logging
import multiprocessing as mp
from typing import Any

import cogment
import numpy as np

from cogment_lab.core import CogmentActor
from cogment_lab.generated import cog_settings


def obs_to_msg(obs: np.ndarray | dict[str, np.ndarray | dict]) -> dict[str, Any]:
    if isinstance(obs, np.ndarray):
        obs = obs.tolist()
    elif isinstance(obs, dict):
        obs = {k: obs_to_msg(v) for k, v in obs.items()}
    elif isinstance(obs, np.integer):
        obs = int(obs)
    elif isinstance(obs, np.floating):
        obs = float(obs)
    return obs


def msg_to_action(data: str, action_map: list[str] | dict[str, int]) -> int:
    if isinstance(action_map, list):
        action_map = {action: i for i, action in enumerate(action_map)}
    if data.startswith("{"):
        action = json.loads(data)
    elif data not in action_map:
        action = action_map["no-op"]
    else:
        action = action_map[data]
    logging.info(f"Processed action {action} from {data} with action_map {action_map}")
    return action


class GradioActor(CogmentActor):
    def __init__(self, send_queue: mp.Queue, recv_queue: mp.Queue):
        super().__init__(send_queue, recv_queue)
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    async def act(self, observation: Any, rendered_frame: np.ndarray | None = None) -> int:
        logging.info(f"Received observation {observation} and frame inside gradio actor")
        obs_data = obs_to_msg(observation)
        self.send_queue.put((obs_data, rendered_frame))
        logging.info(f"Sent observation {obs_data} and frame inside gradio actor")
        action = self.recv_queue.get()
        logging.info(f"Received action {action} inside gradio actor")
        return action


async def run_cogment_actor(port: int, send_queue: asyncio.Queue, recv_queue: asyncio.Queue, signal_queue: mp.Queue):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")
    gradio_actor = GradioActor(send_queue, recv_queue)

    logging.info("Registering actor")
    context.register_actor(impl=gradio_actor.impl, impl_name="gradio", actor_classes=["player"])

    logging.info("Serving actor")
    serve = context.serve_all_registered(cogment.ServedEndpoint(port=port))

    signal_queue.put(True)
    await serve
