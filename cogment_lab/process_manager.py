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
import atexit
import datetime
import logging
import multiprocessing as mp
import os
from asyncio import Task
from collections.abc import Sequence
from multiprocessing import Process, Queue
from typing import Any, Callable, Coroutine

import cogment

from cogment_lab.actors.runner import actor_runner
from cogment_lab.core import BaseActor, BaseEnv
from cogment_lab.envs.runner import env_runner
from cogment_lab.generated import cog_settings, data_pb2
from cogment_lab.humans.runner import human_actor_runner
from cogment_lab.utils.trial_utils import (
    TrialData,
    format_data_multiagent,
    get_actor_params,
)


ORCHESTRATOR_ENDPOINT = "grpc://localhost:9000"
ENVIRONMENT_ENDPOINT = "grpc://localhost:9001"
RANDOM_AGENT_ENDPOINT = "grpc://localhost:9002"
HUMAN_AGENT_ENDPOINT = "grpc://localhost:8999"
DATASTORE_ENDPOINT = "grpc://localhost:9003"


AgentName = str
ImplName = str
TrialName = str


class Cogment:
    """Main Cogment class for managing experiments"""

    def __init__(
        self,
        user_id: str = "cogment_lab",
        torch_mode: bool = False,
        log_dir: str | None = None,
        mp_method: str | None = None,
    ):
        """Initializes the Cogment instance

        Args:
            user_id (str, optional): User ID. Defaults to "cogment_lab".
            torch_mode (bool, optional): Whether to use PyTorch multiprocessing. Defaults to False.
            log_dir (str, optional): Directory to store logs. Defaults to "logs".
            mp_method (str | None, optional): Multiprocessing method to use. Defaults to None.
        """
        self.processes: dict[ImplName, Process] = {}
        self.tasks: dict[ImplName, Task] = {}

        self._register_shutdown_hook()
        self.torch_mode = torch_mode
        self.log_dir = log_dir

        self.envs: dict[ImplName, BaseEnv] = {}
        self.actors: dict[ImplName, BaseActor] = {}

        self.context = cogment.Context(cog_settings=cog_settings, user_id=user_id)
        self.controller = self.context.get_controller(endpoint=cogment.Endpoint(ORCHESTRATOR_ENDPOINT))
        self.datastore = self.context.get_datastore(endpoint=cogment.Endpoint(DATASTORE_ENDPOINT))

        self.env_ports: dict[ImplName, int] = {}
        self.actor_ports: dict[ImplName, int] = {}

        self.mp_ctx = mp.get_context(mp_method) if mp_method else mp.get_context()

        self.trial_envs: dict[TrialName, ImplName] = {}

    def _add_process(
        self,
        target: Callable,
        args: tuple,
        name: ImplName,
        use_torch: bool | None = None,
        force: bool = False,
    ):
        """Adds a process to the list of processes

        Args:
            target (Callable): The process target function
            args (tuple): Arguments for the process target
            name (ImplName): Name of the process
            use_torch (bool | None, optional): Whether to use PyTorch multiprocessing. Defaults to None.
            force (bool, optional): Whether to force adding the process if it already exists. Defaults to False.

        Raises:
            ValueError: If the process already exists and force is False
        """
        if use_torch is None:
            use_torch = self.torch_mode

        if name in self.processes and not force:
            raise ValueError(f"Process {name} already exists")

        if use_torch:
            from torch.multiprocessing import Process as TorchProcess

            p = TorchProcess(target=target, args=args)
        else:
            p = self.mp_ctx.Process(target=target, args=args)
        p.start()
        self.processes[name] = p

    def _add_task(self, target: Coroutine, name: ImplName) -> Task:
        """Adds a task to the list of tasks

        Args:
            target (Coroutine): The task target function
            name (ImplName): Name of the task

        Returns:
            Task: The task instance
        """
        if name in self.tasks:
            raise ValueError(f"Task {name} already exists")

        task = asyncio.create_task(target)
        self.tasks[name] = task
        return task

    def run_env(
        self,
        env: BaseEnv,
        env_name: ImplName,
        port: int = 9001,
        log_file: str | None = None,
    ) -> Coroutine[bool]:
        """Given an environment, runs it in a subprocess

        Args:
            env (BaseEnv): The environment instance
            env_name (ImplName): Name for the environment
            port (int, optional): Port to run the environment on. Defaults to 9001.
            log_file (str | None, optional): Log file path. Defaults to None.

        Returns:
            bool: Whether the environment startup succeeded
        """
        env_class = type(env)
        env_args = env.args
        env_kwargs = env.kwargs

        signal_queue = Queue(1)

        if self.log_dir is not None and log_file:
            log_file = os.path.join(self.log_dir, log_file)

        self._add_process(
            target=env_runner,
            name=env_name,
            args=(
                env_class,
                env_args,
                env_kwargs,
                env_name,
                signal_queue,
                port,
                log_file,
            ),
        )
        logging.info(f"Started environment {env_name} on port {port} with log file {log_file}")

        self.envs[env_name] = env
        self.env_ports[env_name] = port

        return self.is_ready(signal_queue)

    def run_actor(
        self,
        actor: BaseActor,
        actor_name: ImplName,
        port: int = 9002,
        log_file: str | None = None,
    ) -> Coroutine[bool]:
        """Given an actor, runs it

        Args:
            actor (BaseActor): The actor instance
            actor_name (ImplName): Name for the actor
            port (int, optional): Port to run the actor on. Defaults to 9002.
            log_file (str | None, optional): Log file path. Defaults to None.

        Returns:
            bool: Whether the actor startup succeeded
        """
        actor_class = type(actor)
        actor_args = actor.args
        actor_kwargs = actor.kwargs

        signal_queue = Queue(1)

        if self.log_dir is not None and log_file:
            log_file = os.path.join(self.log_dir, log_file)

        self._add_process(
            target=actor_runner,
            name=actor_name,
            args=(
                actor_class,
                actor_args,
                actor_kwargs,
                actor_name,
                signal_queue,
                port,
                log_file,
            ),
        )
        logging.info(f"Started actor {actor_name} on port {port} with log file {log_file}")

        self.actors[actor_name] = actor
        self.actor_ports[actor_name] = port

        return self.is_ready(signal_queue)

    def run_local_actor(
        self,
        actor: BaseActor,
        actor_name: ImplName,
        port: int = 9002,
        log_file: str | None = None,
    ) -> Task:
        """Given an actor, runs it locally

        Args:
            actor (BaseActor): The actor instance
            actor_name (ImplName): Name for the actor
            port (int, optional): Port to run the actor on. Defaults to 9002.
            log_file (str | None, optional): Log file path. Defaults to None.

        Returns:
            bool: Whether the actor startup succeeded
        """

        if self.log_dir is not None and log_file:
            log_file = os.path.join(self.log_dir, log_file)

        self.context.register_actor(impl=actor.impl, impl_name=actor_name, actor_classes=["player"])

        serve = self._add_task(
            self.context.serve_all_registered(cogment.ServedEndpoint(port=port)),
            actor_name,
        )

        logging.info(f"Started actor {actor_name} on port {port} with log file {log_file}")

        self.actors[actor_name] = actor
        self.actor_ports[actor_name] = port

        return serve

    def run_web_ui(
        self,
        app_port: int = 8000,
        cogment_port: int = 8999,
        actions: list[str] | dict[str, Any] = [],
        log_file: str | None = None,
        fps: int = 30,
        html_override: str | None = None,
        file_override: str | None = None,
        jinja_parameters: dict[str, Any] | None = None,
    ) -> Coroutine[bool]:
        """Runs the human actor in a separate process

        Args:
            app_port (int, optional): Port for web UI. Defaults to 8000.
            cogment_port (int, optional): Port for Cogment connection. Defaults to 8999.
            actions (list[str], optional): Allowed actions. Defaults to [].
            log_file (str | None, optional): Log file path. Defaults to None.
            fps (int, optional): Frames per second for environment. Defaults to 30.
            html_override (str | None, optional): HTML override file path. Defaults to None.
            file_override (str | None, optional): File override file path. Defaults to None.
            jinja_parameters (dict[str, Any] | None, optional): Jinja parameters for the HTML override. Defaults to None.

        Returns:
            bool: Whether the web UI startup succeeded
        """

        signal_queue = Queue(1)

        if self.log_dir is not None and log_file:
            log_file = os.path.join(self.log_dir, log_file)

        self._add_process(
            target=human_actor_runner,
            name="web_ui",
            args=(
                app_port,
                cogment_port,
                signal_queue,
                log_file,
                actions,
                fps,
                html_override,
                file_override,
                jinja_parameters,
            ),
        )
        logging.info(f"Started web UI on port {app_port} with log file {log_file}")

        self.actor_ports["web_ui"] = cogment_port

        return self.is_ready(signal_queue)

    def stop_service(self, name: ImplName, timeout: float = 1.0):
        """Stops a process or a task.

        Args:
            name (str): Name of the process or task to stop
            timeout (float, optional): How long (in seconds) to wait for the process to stop before killing it. Defaults to 1.0.
        """
        if name in self.processes:
            self._stop_process(name, timeout)
        elif name in self.tasks:
            self._stop_task(name)
        else:
            raise ValueError(f"Service {name} does not exist")

    def _stop_process(self, name: ImplName, timeout: float = 1.0):
        """Stops a process

        Args:
            name (str): Name of the process to stop
            timeout (float, optional): How long (in seconds) to wait for the process to stop before killing it. Defaults to 1.0.
        """
        if name not in self.processes:
            raise ValueError(f"Process {name} does not exist")
        logging.info(f"Stopping process {name}")
        process = self.processes[name]
        if timeout == 0.0:
            process.kill()
            process.join()
        else:
            process.terminate()
            process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                process.join()
        del self.processes[name]

    def _stop_task(self, name: ImplName):
        """Stops a task

        Args:
            name (str): Name of the task to stop
        """
        if name not in self.tasks:
            raise ValueError(f"Task {name} does not exist")
        logging.info(f"Stopping task {name}")
        task = self.tasks[name]
        task.cancel()
        del self.tasks[name]

    def stop_all_services(self, timeout: float = 1.0):
        """Stops all processes and tasks

        Args:
            timeout (float, optional): How long (in seconds) to wait for the processes to stop before killing them. Defaults to 1.0.
        """
        for name in list(self.processes.keys()):
            self._stop_process(name, timeout)
        for name in list(self.tasks.keys()):
            self._stop_task(name)

    def _cleanup_wrapper(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        asyncio.run(self.cleanup())
        loop.close()

    async def cleanup(self, timeout: float = 1.0):
        """Cleans up all processes. Idempotent."""
        tasks = list(self.tasks.keys())
        processes = list(self.processes.keys())

        for name in tasks:
            try:
                self._stop_task(name)
            except Exception as e:
                logging.warning(f"Failed to stop task {name}: {e}")

        for name in processes:
            try:
                self._stop_process(name, timeout)
            except Exception as e:
                logging.warning(f"Failed to stop process {name}: {e}")

        try:
            await self.context._grpc_server.stop(None)
        except asyncio.exceptions.CancelledError:
            logging.info("Server already stopped")
        except AttributeError:
            logging.info("Server not started")

    def _register_shutdown_hook(self):
        """Registers the cleanup method to run on shutdown"""
        atexit.register(self._cleanup_wrapper)

    async def start_trial(
        self,
        env_name: ImplName,
        actor_impls: dict[AgentName, ImplName] | ImplName,
        session_config: dict[str, Any] = {},
        trial_name: TrialName | None = None,
    ):
        """Starts a new trial

        Args:
            env_name (ImplName): Name of the environment implementation
            actor_impls (dict[AgentName, ImplName] | ImplName): Actor implementations mapped to agent names
            session_config (dict[str, Any]): kwargs for the environment session
            trial_name (str | None, optional): Trial name. Defaults to None.

        Returns:
            str: The trial ID
        """
        if trial_name is None:
            trial_name = f"{env_name}-{datetime.datetime.now().isoformat()}"

        if isinstance(actor_impls, str):
            actor_impls = {"gym": actor_impls}

        env = self.envs[env_name]
        actor_params = [
            get_actor_params(
                name=agent_name,
                implementation=actor_impl,
                agent_specs=env.agent_specs[agent_name],
                endpoint=f"grpc://localhost:{self.actor_ports[actor_impl]}",
            )
            for agent_name, actor_impl in actor_impls.items()
        ]

        env_config = data_pb2.EnvironmentConfig(**session_config)

        trial_params = cogment.TrialParameters(
            cog_settings,
            environment_name=env_name,
            environment_endpoint=f"grpc://localhost:{self.env_ports[env_name]}",
            environment_config=env_config,
            actors=actor_params,
            environment_implementation=env_name,
            datalog_endpoint=DATASTORE_ENDPOINT,
        )

        trial_id = await self.controller.start_trial(trial_id_requested=trial_name, trial_params=trial_params)

        logging.info(f"Started trial {trial_id} with name {trial_name}")

        self.trial_envs[trial_id] = env_name

        return trial_id

    async def get_trial_data(
        self,
        trial_id: str,
        env_name: str | None = None,
        fields: Sequence[str] = (
            "observations",
            "actions",
            "rewards",
            "done",
            "next_observations",
            "last_observation",
        ),
    ) -> dict[str, TrialData]:
        """Gets trial data from the datastore, formatting it appropriately."""
        if env_name is None:
            env_name = self.trial_envs[trial_id]
        env = self.envs[env_name]
        agent_specs = env.agent_specs

        data = await format_data_multiagent(self.datastore, trial_id, agent_specs, fields)

        return data

    async def get_trial(self, trial_id: str):
        """Gets a trial by ID

        Args:
            trial_id (str): The trial ID

        Returns:
            Trial: The trial instance
        """
        [trial] = await self.datastore.get_trials(ids=[trial_id])
        return trial

    def __del__(self):
        """Cleanup on delete"""
        self.stop_all_services()

    async def is_ready(self, queue: Queue):
        """Waits for a readiness signal on a queue

        Args:
            queue (Queue): The queue to wait on

        Returns:
            Any: The object that was put on the queue
        """
        while queue.empty():
            await asyncio.sleep(0.1)
        return queue.get()
