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

import logging
import subprocess

from cogment_lab.constants import COGMENT_LAB_HOME


def launch_service(service_name: str):
    cogment_path = COGMENT_LAB_HOME / "bin/cogment"
    try:
        process = subprocess.Popen([cogment_path, "services", service_name])
        logging.info(f"{service_name} launched successfully. PID: {process.pid}")
        return process
    except Exception as e:
        logging.error(f"Failed to launch {service_name}: {e}")
        return None


def launch_main(command: str):
    services = command.split()
    processes = []

    if "base" in services or "all" in services:
        if len(services) > 1:
            raise ValueError("Cannot combine 'base' or 'all' with other services")

    if "base" in services:
        services_to_run = ["orchestrator", "trial_datastore"]
    elif "all" in services:
        services_to_run = [
            "orchestrator",
            "trial_datastore",
            "model_registry",
            "directory",
            "web_proxy",
        ]
    else:
        services_to_run = services

    for service in services_to_run:
        process = launch_service(service)
        if process:
            processes.append(process)

    # Optional: Wait for all subprocesses to complete
    for process in processes:
        process.wait()

    return processes
