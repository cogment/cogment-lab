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
import signal
from multiprocessing import Queue
from typing import Any

from cogment_lab.humans.actor import run_cogment_actor, start_fastapi
from cogment_lab.utils.runners import setup_logging


# def human_actor_runner(
#     app_port: int = 8000,
#     cogment_port: int = 8999,
#     log_file: str | None = None
# ):
#     """Runs the human actor along with the FastAPI server"""
#     if log_file:
#         setup_logging(log_file)
#
#     # Queues for communication between FastAPI and Cogment actor
#     app_to_actor = asyncio.Queue()
#     actor_to_app = asyncio.Queue()
#
#     # Asyncio tasks for the FastAPI server and Cogment actor
#     fastapi_task = start_fastapi(port=app_port, send_queue=app_to_actor, recv_queue=actor_to_app)
#     cogment_task = asyncio.create_task(run_cogment_actor(port=cogment_port, send_queue=actor_to_app, recv_queue=app_to_actor))
#
#     # Run the asyncio event loop
#     asyncio.run(asyncio.gather(fastapi_task, cogment_task))


async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


def signal_handler(sig, frame):
    asyncio.create_task(shutdown())


async def human_actor_main(
    app_port: int,
    cogment_port: int,
    signal_queue: Queue,
    actions: list[str] | None = None,
    fps: float = 30,
    html_override: str | None = None,
    file_override: str | None = None,
    jinja_parameters: dict[str, Any] | None = None,
):
    app_to_actor = asyncio.Queue()
    actor_to_app = asyncio.Queue()
    fastapi_task = asyncio.create_task(
        start_fastapi(
            port=app_port,
            send_queue=app_to_actor,
            recv_queue=actor_to_app,
            actions=actions,
            fps=fps,
            html_override=html_override,
            file_override=file_override,
            jinja_parameters=jinja_parameters,
        )
    )
    cogment_task = asyncio.create_task(
        run_cogment_actor(
            port=cogment_port,
            send_queue=actor_to_app,
            recv_queue=app_to_actor,
            signal_queue=signal_queue,
        )
    )

    await asyncio.gather(fastapi_task, cogment_task)


def human_actor_runner(
    app_port: int,
    cogment_port: int,
    signal_queue: Queue,
    log_file: str | None = None,
    actions: list[str] | None = None,
    fps: float = 30,
    html_override: str | None = None,
    file_override: str | None = None,
    jinja_parameters: dict[str, Any] | None = None,
):
    if log_file:
        setup_logging(log_file)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda s=sig, frame=None: signal_handler(s, frame))

    try:
        loop.run_until_complete(
            human_actor_main(
                app_port,
                cogment_port,
                signal_queue,
                actions,
                fps,
                html_override,
                file_override,
                jinja_parameters,
            )
        )
    finally:
        loop.close()
