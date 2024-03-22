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

import asyncio
import json
import logging
import multiprocessing as mp
import signal
from typing import Any, Callable

import cogment
import numpy as np

from cogment_lab.core import CogmentActor
from cogment_lab.generated import cog_settings
from cogment_lab.utils.runners import setup_logging


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
        # logging.info(f"Received observation {observation} and frame inside gradio actor")
        obs_data = obs_to_msg(observation)
        self.send_queue.put((obs_data, rendered_frame))
        # logging.info(f"Sent observation {obs_data} and frame inside gradio actor")
        action = self.recv_queue.get()
        # logging.info(f"Received action {action} inside gradio actor")
        return action

    async def on_ending(self, observation, rendered_frame):
        obs_data = obs_to_msg(observation)
        self.send_queue.put((obs_data, rendered_frame))


async def run_cogment_actor(port: int, send_queue: asyncio.Queue, recv_queue: asyncio.Queue, signal_queue: mp.Queue):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")
    gradio_actor = GradioActor(send_queue, recv_queue)

    logging.info("Registering actor")
    context.register_actor(impl=gradio_actor.impl, impl_name="gradio", actor_classes=["player"])

    logging.info("Serving actor")
    serve = context.serve_all_registered(cogment.ServedEndpoint(port=port))

    signal_queue.put(True)
    await serve


async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


def signal_handler(sig, frame):
    asyncio.create_task(shutdown())


async def gradio_actor_main(
    cogment_port: int,
    gradio_app_fn: Callable[[mp.Queue, mp.Queue, str], None],
    signal_queue: mp.Queue,
    log_file: str | None = None,
):
    gradio_to_actor = mp.Queue()
    actor_to_gradio = mp.Queue()

    logging.info("Starting gradio interface")
    process = mp.Process(target=gradio_app_fn, args=(gradio_to_actor, actor_to_gradio, log_file))
    process.start()

    try:
        logging.info("Starting cogment actor")
        cogment_task = asyncio.create_task(
            run_cogment_actor(
                port=cogment_port,
                send_queue=actor_to_gradio,
                recv_queue=gradio_to_actor,
                signal_queue=signal_queue,
            )
        )

        logging.info("Waiting for cogment actor to finish")

        await cogment_task
    finally:
        process.terminate()
        process.join()


def gradio_actor_runner(
    cogment_port: int,
    gradio_app_fn: Callable[[mp.Queue, mp.Queue, str], None],
    signal_queue: mp.Queue,
    log_file: str | None = None,
):
    if log_file:
        setup_logging(log_file)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda s=sig, frame=None: signal_handler(s, frame))

    try:
        loop.run_until_complete(
            gradio_actor_main(
                cogment_port=cogment_port,
                gradio_app_fn=gradio_app_fn,
                signal_queue=signal_queue,
                log_file=log_file,
            )
        )
    finally:
        loop.run_until_complete(shutdown())
        loop.close()
