import numpy as np
import PIL
import pytest

from cogment_lab.specs.encode_rendered_frame import (
    decode_rendered_frame,
    encode_rendered_frame,
)


def create_test_image(width, height):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_encode_valid_input():
    test_image = create_test_image(100, 100)
    encoded = encode_rendered_frame(test_image)
    assert isinstance(encoded, bytes)


def test_encode_invalid_size():
    test_image = create_test_image(100, 100)
    encoded = encode_rendered_frame(test_image, -1)
    decoded = decode_rendered_frame(encoded)
    assert max(decoded.shape[:2]) == 100


def test_encode_with_resizing():
    test_image = create_test_image(2000, 2000)
    max_size = 500
    encoded = encode_rendered_frame(test_image, max_size)
    decoded = decode_rendered_frame(encoded)

    assert max(decoded.shape[:2]) == max_size

    original_aspect_ratio = test_image.shape[1] / test_image.shape[0]
    decoded_aspect_ratio = decoded.shape[1] / decoded.shape[0]
    np.testing.assert_almost_equal(original_aspect_ratio, decoded_aspect_ratio)


def test_decode_valid_input():
    test_image = create_test_image(100, 100)
    encoded = encode_rendered_frame(test_image)
    decoded = decode_rendered_frame(encoded)
    assert decoded.shape == test_image.shape


def test_decode_failure():
    with pytest.raises(PIL.UnidentifiedImageError):
        decode_rendered_frame(b"invalid data")


def test_roundtrip():
    test_image = create_test_image(100, 100)
    encoded = encode_rendered_frame(test_image)
    decoded = decode_rendered_frame(encoded)
    np.testing.assert_array_almost_equal(test_image, decoded)
