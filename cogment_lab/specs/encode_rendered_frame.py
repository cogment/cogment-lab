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

import cv2
import numpy as np

MAX_RENDERED_WIDTH = 2048


def encode_rendered_frame(rendered_frame: np.ndarray, max_size: int = MAX_RENDERED_WIDTH) -> bytes:
    if max_size <= 0:
        max_size = MAX_RENDERED_WIDTH
    # gRPC max message size hack
    height, width = rendered_frame.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(new_height / height * width)
        else:
            new_width = max_size
            new_height = int(height / width * new_width)
        rendered_frame = cv2.resize(rendered_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # note rgb -> bgr for cv2
    result, encoded_frame = cv2.imencode(".jpg", rendered_frame[:, :, ::-1])
    assert result

    return encoded_frame.tobytes()


def decode_rendered_frame(encoded_frame: bytes) -> np.ndarray:
    """
    Decode the rendered frame from bytes to a NumPy array.

    Args:
        encoded_frame (bytes): The encoded frame as a byte array.

    Returns:
        np.ndarray: The decoded rendered frame as a NumPy array.
    """
    if encoded_frame is None or len(encoded_frame) == 0:
        return None

    # Convert the byte array back to a NumPy array
    encoded_frame_np = np.frombuffer(encoded_frame, dtype=np.uint8)

    # Decode the image from the byte array
    decoded_frame = cv2.imdecode(encoded_frame_np, cv2.IMREAD_COLOR)

    # Check if the decoding was successful
    if decoded_frame is None:
        raise ValueError("Failed to decode the rendered frame.")

    # Convert from BGR to RGB
    decoded_frame = decoded_frame[:, :, ::-1]

    return decoded_frame
