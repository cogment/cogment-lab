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

import logging
import os
import subprocess
import sys


def setup_logging(log_file: str):
    """
    Set up logging to file and stdout/stderr.

    Args:
        log_file: Path to log file
    """
    # Redirect stdout and stderr to log file
    dirname = os.path.dirname(log_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(log_file, "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],  # , logging.StreamHandler()],
    )


def process_cleanup():
    """
    Clean up any leftover processes related to cogment_lab.
    """
    try:
        pid = os.getpid()

        command = (
            f"ps aux | grep 'cogment-lab' | grep 'multiprocessing' | grep -v grep | "
            f"awk '{{if ($3 != {pid}) print $2}}' | xargs -r kill -9"
        )

        subprocess.run(command, shell=True, check=True)
        print("Processes terminated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
