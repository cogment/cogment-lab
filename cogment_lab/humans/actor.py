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
import base64
import io
import json
import logging
import multiprocessing as mp
import os
from typing import Any

import cogment
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
from PIL import Image

from cogment_lab.core import CogmentActor
from cogment_lab.generated import cog_settings


def image_to_msg(img: np.ndarray | None) -> str | None:
    if img is None:
        return None
    image = Image.fromarray(img)
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="PNG")  # type: ignore
    base64_encoded_result_bytes = base64.b64encode(img_byte_array.getvalue())
    base64_encoded_result_str = base64_encoded_result_bytes.decode("ascii")
    return f"data:image/png;base64,{base64_encoded_result_str}"


def msg_to_action(data: str, action_map: list[str] | dict[str, int]) -> int:
    if isinstance(action_map, list):
        action_map = {action: i for i, action in enumerate(action_map)}

    if data.startswith("{"):
        # This is a JSON object
        action = json.loads(data)
    elif data not in action_map:
        action = action_map["no-op"]
    else:
        action = action_map[data]

    logging.info(f"Processed action {action} from {data} with action_map {action_map}")
    return action


async def start_fastapi(
    port: int,
    send_queue: asyncio.Queue,
    recv_queue: asyncio.Queue,
    actions: list[str] | dict[str, Any] | None = None,
    fps: float = 30.0,
    html_override: str | None = None,
    file_override: str | None = None,
    jinja_parameters: dict[str, Any] | None = None,
):
    app = FastAPI()

    if actions is None:
        actions = ["no-op", "ArrowLeft", "ArrowRight"]

    if jinja_parameters is None:
        jinja_parameters = {}

    @app.get("/")
    async def get():
        logging.info("Serving index.html")
        if html_override is not None:
            # Render HTML from string
            template = Template(html_override)
            rendered_html = template.render(**jinja_parameters)
            return HTMLResponse(rendered_html)
        elif file_override is not None and os.path.isfile(file_override):
            # Render HTML from file
            with open(file_override) as file:
                file_content = file.read()
            template = Template(file_content)
            rendered_html = template.render(**jinja_parameters)
            return HTMLResponse(rendered_html)
        else:
            # Fallback option: Serve static file
            static_directory_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static")
            return FileResponse(os.path.join(static_directory_path, "index.html"))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        logging.info("Waiting for socket connection")
        last_action_data = "no-op"
        logging.info(f"Set {last_action_data=}")
        await websocket.accept()
        logging.info("Client connected")
        while True:
            try:
                logging.info("Waiting for frame")
                frame: np.ndarray = await recv_queue.get()
                if not isinstance(frame, np.ndarray):
                    logging.warning(f"Got frame of type {type(frame)}")
                    continue
                logging.info(f"Got frame with shape {frame.shape}")
                msg = image_to_msg(frame)
                if msg is not None:
                    await websocket.send_text(msg)

                try:
                    action_data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0 / fps)
                    last_action_data = action_data
                    logging.info(f"Got action {action_data}, updated {last_action_data=}")
                except asyncio.TimeoutError:
                    logging.info(f"Timed out waiting for action, using {last_action_data=}")
                    action_data = last_action_data

                action = msg_to_action(action_data, actions)

                await send_queue.put(action)
            except WebSocketDisconnect:
                logging.info("Client disconnected, waiting for new connection.")
                await websocket.close()
                await websocket.accept()  # Accept a new WebSocket connection
            except Exception as e:
                logging.error("An error occurred: %s", e)
                break  # Break the loop in case of non-WebSocketDisconnect exceptions

    current_file_path = os.path.abspath(os.path.dirname(__file__))
    static_directory_path = os.path.join(current_file_path, "static")
    app.mount("/static", StaticFiles(directory=static_directory_path), name="static")

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)

    await server.serve()


class HumanPlayer(CogmentActor):
    def __init__(self, send_queue: asyncio.Queue, recv_queue: asyncio.Queue):
        super().__init__(send_queue, recv_queue)
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    async def act(self, observation: Any, rendered_frame: np.ndarray | None = None) -> int:
        # logging.info(
        #     f"Getting an action with {observation=}" + f" and {rendered_frame.shape=}"
        #     if rendered_frame is not None
        #     else "no frame"
        # )
        await self.send_queue.put(rendered_frame)
        action = await self.recv_queue.get()
        return action


async def run_cogment_actor(
    port: int,
    send_queue: asyncio.Queue,
    recv_queue: asyncio.Queue,
    signal_queue: mp.Queue,
):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")

    human_player = HumanPlayer(send_queue, recv_queue)

    logging.info("Registering actor")

    context.register_actor(impl=human_player.impl, impl_name="web_ui", actor_classes=["player"])
    logging.info("Serving actor")

    serve = context.serve_all_registered(cogment.ServedEndpoint(port=port))

    signal_queue.put(True)

    await serve
