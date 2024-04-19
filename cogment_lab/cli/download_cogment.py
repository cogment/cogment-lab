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

import json
import os
import platform
import re
import stat
from enum import Enum
from tempfile import mkdtemp
from urllib.request import urlopen, urlretrieve


class Arch(Enum):
    AMD64 = "amd64"
    ARM64 = "arm64"


def get_current_arch() -> Arch:
    py_machine = platform.machine()
    if py_machine in ["x86_64", "i686", "AMD64", "aarch64"]:
        return Arch.AMD64

    if py_machine in ["arm64"]:
        return Arch.ARM64

    raise RuntimeError(f"Unsupported architecture [{py_machine}]")


class Os(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"


def get_current_os() -> Os:
    py_system = platform.system()
    if py_system == "Darwin":
        return Os.MACOS
    if py_system == "Windows":
        return Os.WINDOWS
    if py_system == "Linux":
        return Os.LINUX

    raise RuntimeError(f"Unsupported os [{py_system}]")


def get_latest_release_version() -> str:
    res = urlopen("https://api.github.com/repos/cogment/cogment/releases/latest")

    parsedBody = json.load(res)

    return parsedBody["tag_name"]


def download_cogment(
    output_dir: str | None = None,
    desired_version: str | None = None,
    desired_arch: Arch | None = None,
    desired_os: Os | None = None,
):
    """
    Download a version of cogment

    Parameters:
    - output_dir (string, optional): the output directory, if undefined a temporary directory will be used.
    - desired_version (string, optional): the desired version,
      if undefined the latest released version (excluding prereleases) will be used.
    - desired_arch (Arch, optional): the desired architecture,
      if undefined the current architecture will be detected and used.
    - os (Os, optional): the desired os, if undefined the current os will be detected and used.

    Returns:
        path to the downloaded cogment
    """
    if not output_dir:
        output_dir = mkdtemp()
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if not desired_version:
        desired_version = get_latest_release_version()

    try:
        desired_version = re.findall(r"[0-9]+.[0-9]+.[0-9]+(?:-[a-zA-Z0-9]+)?", desired_version)[0]
    except RuntimeError as exc:
        raise RuntimeError(f"Desired cogment version [{desired_version}] doesn't follow the expected patterns") from exc

    if desired_arch is None:
        desired_arch = get_current_arch()

    if desired_os is None:
        desired_os = get_current_os()

    cogment_url = (
        "https://github.com/cogment/cogment/releases/download/"
        + f"v{desired_version}/cogment-{desired_os.value}-{desired_arch.value}"
    )

    cogment_filename = os.path.join(output_dir, "cogment")
    if desired_os == Os.WINDOWS:
        cogment_url += ".exe"
        cogment_filename += ".exe"

    try:
        cogment_filename, _ = urlretrieve(cogment_url, cogment_filename)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to retrieve cogment version [{desired_version}] for arch "
            + f"[{desired_arch}] and os [{desired_os}] from [{cogment_url}] to [{cogment_filename}]"
        ) from exc

    # Make sure it is executable
    cogment_stat = os.stat(cogment_filename)
    os.chmod(cogment_filename, cogment_stat.st_mode | stat.S_IEXEC)

    return cogment_filename
