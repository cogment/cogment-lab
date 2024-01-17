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
from multiprocessing import Queue

import cogment

from cogment_lab.generated import cog_settings
from cogment_lab.utils.runners import setup_logging
from cogment_lab.core import BaseActor


async def register_actor(actor: BaseActor, actor_name: str, queue: Queue, port: int = 9002):
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_lab")

    context.register_actor(impl=actor.impl, impl_name=actor_name, actor_classes=["player"])

    serve = context.serve_all_registered(cogment.ServedEndpoint(port=port))

    queue.put(True)

    await serve


def actor_runner(
    actor_class: type,
    actor_args: tuple,
    actor_kwargs: dict,
    actor_name: str,
    signal_queue: Queue,
    port: int = 9002,
    log_file: str | None = None,
):
    """Given an actor, runs it"""
    if log_file:
        setup_logging(log_file)
    actor = actor_class(*actor_args, **actor_kwargs)

    asyncio.run(register_actor(actor, actor_name, signal_queue, port))
