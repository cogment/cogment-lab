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

import argparse
import logging
import os
import subprocess
import sys


TEAL = "\033[36m"
RESET = "\033[0m"

custom_format = f"%(asctime)s {TEAL}[%(levelname)s] [%(name)s]{RESET} %(message)s [thread:%(thread)d]"

formatter = logging.Formatter(custom_format, datefmt="%Y-%m-%dT%H:%M:%S%z")

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])

sys.path.insert(0, "..")


def install_cogment(path: str | None = None):
    try:
        subprocess.run(
            [
                "curl",
                "--silent",
                "-L",
                "https://raw.githubusercontent.com/cogment/cogment/main/install.sh",
                "--output",
                "install-cogment.sh",
            ],
            check=True,
        )
        subprocess.run(["chmod", "+x", "install-cogment.sh"], check=True)
        cmd = ["sudo", "./install-cogment.sh"]
        if path:
            cmd += ["--install-dir", path]
        cmd += ["--version", "2.19.1"]
        if os.getenv("GITHUB_ACTIONS") == "true":
            cmd = cmd[1:]  # Remove sudo for github actions
        subprocess.run(cmd, check=True)
        logging.info("Cogment installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Installation failed: {e}")
    finally:
        if os.path.exists("install-cogment.sh"):
            os.remove("install-cogment.sh")
            logging.info("Cleanup completed.")


def main():
    parser = argparse.ArgumentParser(description="Cogment Lab CLI")
    subparsers = parser.add_subparsers(dest="command")

    # launch subcommand
    parser_launch = subparsers.add_parser("launch")
    parser_launch.add_argument("file")

    # env subcommand
    parser_env = subparsers.add_parser("env")
    parser_env.add_argument("file")

    parser_actor = subparsers.add_parser("actor")
    parser_actor.add_argument("file")

    parser_install = subparsers.add_parser("install")
    parser_install.add_argument("path", nargs="?")

    args = parser.parse_args()

    if args.command == "install":
        install_cogment(args.path)
    elif args.command == "launch":
        from cogment_lab.cli import launch

        launch.launch_main(args.file)
    elif args.command == "env":
        from cogment_lab.cli import env

        env.env_main(args.file)
    elif args.command == "actor":
        from cogment_lab.cli import actor

        actor.actor_main(args.file)
    else:
        print("Invalid command. Use 'launch', 'env', 'actor' or `install`.")


if __name__ == "__main__":
    main()
