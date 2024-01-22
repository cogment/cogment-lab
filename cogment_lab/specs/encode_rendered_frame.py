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

import io

import numpy as np
from PIL import Image


MAX_RENDERED_WIDTH = 2048


def encode_rendered_frame(rendered_frame: np.ndarray, max_size: int = MAX_RENDERED_WIDTH, format: str = "PNG") -> bytes:
    if max_size <= 0:
        max_size = MAX_RENDERED_WIDTH

    image = Image.fromarray(rendered_frame.astype("uint8"), "RGB")

    width, height = image.size
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(new_height / height * width)
        else:
            new_width = max_size
            new_height = int(height / width * new_width)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

    with io.BytesIO() as output:
        image.save(output, format=format)
        encoded_frame = output.getvalue()

    return encoded_frame


def decode_rendered_frame(encoded_frame: bytes) -> np.ndarray:
    assert len(encoded_frame) > 0, "Encoded frame is empty"

    with io.BytesIO(encoded_frame) as input:
        image = Image.open(input)
        decoded_frame = np.array(image)

    if decoded_frame is None:
        raise ValueError("Failed to decode the rendered frame.")

    return decoded_frame
